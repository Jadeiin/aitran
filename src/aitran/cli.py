"""CLI entry point using click."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from importlib.resources import files
from threading import Lock
from typing import TYPE_CHECKING

import click
from crowdin_api.exceptions import CrowdinException
from requests import RequestException
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from wlc.client import WeblateException

from aitran.crowdin import download_translation as crowdin_download_translation
from aitran.crowdin import get_progress as crowdin_get_progress
from aitran.crowdin import list_files as crowdin_list_files
from aitran.crowdin import list_languages as crowdin_list_languages
from aitran.crowdin import list_projects as crowdin_list_projects
from aitran.crowdin import upload_translation as crowdin_upload_translation
from aitran.headless import review_file, translate_directory, translate_file
from aitran.manipulate import remove_by_options
from aitran.observability import (
    ObservabilityError,
    flush_logfire,
    flush_mlflow,
    setup_logfire,
    setup_mlflow,
)
from aitran.sync import sync
from aitran.utils import (
    copy_file_if_not_exists,
    find_config,
    open_file_by_default,
    open_file_explorer,
)
from aitran.weblate import download_translation as weblate_download_translation
from aitran.weblate import get_stats as weblate_get_stats
from aitran.weblate import list_objects as weblate_list_objects
from aitran.weblate import upload_translation as weblate_upload_translation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from aitran.agent_work import BatchProgress

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
WEBLATE_UPLOAD_METHODS = [
    "translate",
    "approve",
    "suggest",
    "fuzzy",
    "replace",
    "source",
    "add",
]
WEBLATE_FUZZY_CHOICES = ["process", "approve"]
WEBLATE_CONFLICT_CHOICES = ["ignore", "replace-translated", "replace-approved"]
WEBLATE_DOWNLOAD_FORMATS = ["po", "xliff11", "xliff"]


@contextmanager
def _progress_reporter() -> Iterator[Callable[[str, BatchProgress], None]]:
    """Render all headless Agent files in one thread-safe Rich display.

    Yields:
        Callback suitable for the headless Agent adapters.
    """
    lock = Lock()
    task_ids: dict[str, int] = {}
    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
        console=Console(stderr=True),
    )
    progress.live.vertical_overflow = "crop"

    def report(path: str, update: BatchProgress) -> None:
        with lock:
            task_id = task_ids.get(path)
            if task_id is None:
                task_id = progress.add_task(path, total=update.total)
                task_ids[path] = task_id
            progress.update(
                task_id,
                completed=update.accepted,
                total=update.total,
                refresh=True,
            )

    with progress:
        yield report


def _weblate_auth_options(command):
    """Apply common Weblate connection options.

    Args:
        command: Click command function.

    Returns:
        Decorated command function.
    """
    command = click.option(
        "--token",
        envvar="AITRAN_WEBLATE_TOKEN",
        required=True,
        help="Weblate API token",
    )(command)
    return click.option(
        "--url",
        envvar="AITRAN_WEBLATE_URL",
        required=True,
        help="Weblate base URL (e.g. https://weblate.example.org)",
    )(command)


def _crowdin_auth_options(command):
    """Apply common Crowdin connection options.

    Args:
        command: Click command function.

    Returns:
        Decorated command function.
    """
    command = click.option(
        "--base-url",
        envvar="AITRAN_CROWDIN_BASE_URL",
        help="Crowdin API base URL override",
    )(command)
    command = click.option(
        "--organization",
        envvar="AITRAN_CROWDIN_ORG",
        help="Crowdin organization (Enterprise only)",
    )(command)
    return click.option(
        "--token",
        envvar="AITRAN_CROWDIN_TOKEN",
        required=True,
        help="Crowdin API token",
    )(command)


def _crowdin_project_options(command):
    """Apply common Crowdin project selection options.

    Args:
        command: Click command function.

    Returns:
        Decorated command function.
    """
    command = click.option("--project", "project_name", help="Crowdin project name")(
        command
    )
    return click.option("--project-id", type=int, help="Crowdin project ID")(command)


def _timeout_option(command):
    """Apply the common API timeout option.

    Args:
        command: Click command function.

    Returns:
        Decorated command function.
    """
    return click.option(
        "--timeout",
        "timeout_seconds",
        type=click.IntRange(min=1),
        default=120,
        show_default=True,
        help="Timeout (seconds) for API operations",
    )(command)


def _observability_options(command):
    """Apply common observability options (Logfire + MLflow).

    Args:
        command: Click command function.

    Returns:
        Decorated command function.
    """
    for opt in [
        click.option(
            "--mlflow-experiment",
            envvar="AITRAN_MLFLOW_EXPERIMENT",
            help="MLflow experiment name (defaults to 'Default').",
        ),
        click.option(
            "--mlflow-tracking-uri",
            envvar="AITRAN_MLFLOW_TRACKING_URI",
            help="MLflow tracking server URI (defaults to local ./mlruns).",
        ),
        click.option(
            "--mlflow",
            is_flag=True,
            envvar="AITRAN_MLFLOW",
            help=(
                "Enable MLflow tracing for agent/model runs. "
                "Prompts and completions may be logged to MLflow."
            ),
        ),
        click.option(
            "--logfire-capture-http",
            is_flag=True,
            envvar="AITRAN_LOGFIRE_CAPTURE_HTTP",
            help=(
                "Also capture provider HTTP headers and bodies in Logfire. "
                "This may include prompts, completions, and credentials."
            ),
        ),
        click.option(
            "--logfire",
            is_flag=True,
            envvar="AITRAN_LOGFIRE",
            help=(
                "Enable Pydantic Logfire tracing for agent/model runs. "
                "Prompts and completions may be sent to Logfire."
            ),
        ),
    ]:
        command = opt(command)
    return command


@contextmanager
def _observability(
    *, logfire, logfire_capture_http, mlflow, mlflow_tracking_uri, mlflow_experiment
):
    """Set up and tear down observability backends.

    Raises:
        click.ClickException: If an observability backend cannot be configured.
    """
    logfire_enabled = False
    mlflow_enabled = False
    try:
        logfire_enabled = setup_logfire(
            enabled=logfire,
            capture_http=logfire_capture_http,
        )
        mlflow_enabled = setup_mlflow(
            enabled=mlflow,
            tracking_uri=mlflow_tracking_uri,
            experiment=mlflow_experiment,
        )
    except ObservabilityError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        yield
    finally:
        flush_logfire(enabled=logfire_enabled)
        flush_mlflow(enabled=mlflow_enabled)


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.version_option(message="%(prog)s %(version)s")
@click.option(
    "-m",
    "--model",
    "orchestrator_model",
    envvar="AITRAN_APP_MODEL",
    default="deepseek:deepseek-v4-pro",
    show_default=True,
    help=("Model for the orchestrator agent (provider:model format)."),
)
@click.option(
    "-k",
    "--key",
    "orchestrator_key",
    envvar="AITRAN_APP_KEY",
    help="API key for the orchestrator model",
)
@click.option(
    "--host",
    "orchestrator_host",
    envvar="AITRAN_APP_HOST",
    help="Custom API base URL for the orchestrator model",
)
@click.option(
    "--temperature",
    "orchestrator_temperature",
    envvar="AITRAN_APP_TMP",
    type=float,
    default=0.5,
    show_default=True,
    help="LLM temperature for the orchestrator model.",
)
@click.option(
    "--crowdin-token",
    envvar="AITRAN_CROWDIN_TOKEN",
    help="Crowdin API token",
)
@click.option(
    "--crowdin-org",
    envvar="AITRAN_CROWDIN_ORG",
    help="Crowdin organization (Enterprise only)",
)
@click.option(
    "--crowdin-url",
    envvar="AITRAN_CROWDIN_BASE_URL",
    help="Crowdin API base URL override",
)
@click.option(
    "--weblate-url",
    envvar="AITRAN_WEBLATE_URL",
    help="Weblate base URL",
)
@click.option(
    "--weblate-token",
    envvar="AITRAN_WEBLATE_TOKEN",
    help="Weblate API token",
)
@click.option(
    "--translate-model",
    envvar=["AITRAN_TRANSLATOR_MODEL", "AITRAN_MODEL"],
    default="deepseek:deepseek-v4-flash",
    show_default=True,
    help="Model for translation tasks",
)
@click.option(
    "--translate-key",
    envvar=["AITRAN_TRANSLATOR_KEY", "AITRAN_API_KEY"],
    help="API key for the translation model",
)
@click.option(
    "--translate-host",
    envvar=["AITRAN_TRANSLATOR_HOST", "AITRAN_API_HOST"],
    help="Custom API base URL for the translation model",
)
@click.option(
    "--translate-temperature",
    envvar=["AITRAN_TRANSLATOR_TMP", "AITRAN_MODEL_TMP"],
    type=float,
    default=0.1,
    show_default=True,
    help="LLM temperature for translation tasks.",
)
@click.option(
    "--review-model",
    envvar="AITRAN_REVIEWER_MODEL",
    help="Reviewer model; defaults to the translator model.",
)
@click.option(
    "--review-key",
    envvar="AITRAN_REVIEWER_KEY",
    help="Reviewer API key; defaults to the translator key.",
)
@click.option(
    "--review-host",
    envvar="AITRAN_REVIEWER_HOST",
    help="Reviewer API host; defaults to the translator host.",
)
@click.option(
    "--review-temperature",
    envvar="AITRAN_REVIEWER_TMP",
    type=float,
    help="Reviewer temperature; defaults to the translator temperature.",
)
@click.option(
    "--web-host",
    envvar="AITRAN_WEB_HOST",
    default="127.0.0.1",
    show_default=True,
    help="Host for the orchestrator WebUI server.",
)
@click.option(
    "--web-port",
    envvar="AITRAN_WEB_PORT",
    type=click.IntRange(min=1, max=65535),
    default=7932,
    show_default=True,
    help="Port for the orchestrator WebUI server.",
)
@_observability_options
@click.pass_context
def app(
    ctx: click.Context,
    orchestrator_model: str,
    orchestrator_key: str | None,
    orchestrator_host: str | None,
    orchestrator_temperature: float,
    crowdin_token: str | None,
    crowdin_org: str | None,
    crowdin_url: str | None,
    weblate_url: str | None,
    weblate_token: str | None,
    translate_model: str,
    translate_key: str | None,
    translate_host: str | None,
    translate_temperature: float,
    review_model: str | None,
    review_key: str | None,
    review_host: str | None,
    review_temperature: float | None,
    web_host: str,
    web_port: int,
    logfire: bool,
    logfire_capture_http: bool,
    mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str | None,
) -> None:
    """Aitran — Translate PO and XLIFF files using LLMs.

    Built on Pydantic AI. Supports OpenAI, Anthropic, and any OpenAI-compatible
    provider; specify models as `<provider>:<model>` (e.g. `openai:gpt-5.4-mini`,
    `anthropic:claude-haiku-4-5`).
    """
    if ctx.invoked_subcommand is not None:
        return

    from aitran.app import run_web_app
    from aitran.toolsets._base import OrchestratorDeps

    with _observability(
        logfire=logfire,
        logfire_capture_http=logfire_capture_http,
        mlflow=mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    ):
        deps = OrchestratorDeps(
            crowdin_token=crowdin_token,
            crowdin_organization=crowdin_org,
            crowdin_base_url=crowdin_url,
            weblate_url=weblate_url,
            weblate_token=weblate_token,
            translate_model=translate_model,
            translate_api_key=translate_key,
            translate_api_host=translate_host,
            translate_temperature=translate_temperature,
            review_model=review_model,
            review_api_key=review_key,
            review_api_host=review_host,
            review_temperature=review_temperature,
        )
        run_web_app(
            orchestrator_model=orchestrator_model,
            orchestrator_api_key=orchestrator_key,
            orchestrator_api_host=orchestrator_host,
            orchestrator_temperature=orchestrator_temperature,
            deps=deps,
            web_host=web_host,
            web_port=web_port,
        )


@app.command(
    "translate", context_settings=CONTEXT_SETTINGS, help="Translate PO/XLIFF files."
)
@click.option(
    "-m",
    "--model",
    envvar="AITRAN_MODEL",
    default="deepseek:deepseek-v4-flash",
    help=(
        "Model in <provider>:<model> format "
        "(e.g. openai:gpt-5.4-mini, anthropic:claude-haiku-4-5)"
    ),
)
@click.option(
    "-k", "--key", envvar="AITRAN_API_KEY", help="API key for the LLM provider"
)
@click.option("--host", envvar="AITRAN_API_HOST", help="Custom API base URL")
@click.option(
    "-t",
    "--temperature",
    envvar="AITRAN_MODEL_TMP",
    type=float,
    default=0.1,
    help="LLM temperature (0.0-2.0)",
)
@click.option("--po", "po_file", type=click.Path(exists=True), help="PO file path")
@click.option(
    "--po-dir",
    "po_dir",
    type=click.Path(exists=True, file_okay=False),
    help="Directory of .po files",
)
@click.option(
    "--xliff",
    "xliff_file",
    type=click.Path(exists=True),
    help="XLIFF file path",
)
@click.option(
    "--xliff-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Directory of .xliff/.xlf files",
)
@click.option("-src", "--source", default="en", help="Source language (ISO 639-1)")
@click.option("-l", "--lang", help="Target language (ISO 639-1)")
@click.option("-v", "--verbose", is_flag=True, help="Print artifact JSON")
@click.option(
    "--context",
    "context_file",
    type=click.Path(exists=True),
    help="Text file with additional translation context",
)
@click.option(
    "--max-page-units",
    "--batch-size",
    "max_page_units",
    type=click.IntRange(min=1),
    default=100,
    help="Max units per translator retrieval page",
)
@click.option(
    "--max-page-chars",
    type=click.IntRange(min=1),
    default=4096,
    show_default=True,
    help="Max source/target characters per translator retrieval page",
)
@click.option(
    "--scope",
    type=click.Choice(["pending", "repair", "all"]),
    default="pending",
    show_default=True,
    help="Translation selection scope",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Max files to translate concurrently for directory inputs",
)
@click.option(
    "--order",
    type=click.Choice(["file", "source", "reference", "context"]),
    default="file",
    show_default=True,
    help=(
        "Translation unit ordering: file (preserve original order), "
        "source (sort alphabetically for dedup), "
        "reference (group by source file path), "
        "context (group by msgctxt)"
    ),
)
@click.option(
    "--profile",
    "profile",
    type=click.Choice(["fast", "full"]),
    default="full",
    show_default=True,
    help=(
        "Prompt detail level: fast (minimal unit fields), "
        "full (all metadata: context, location, note, flag)"
    ),
)
@click.option(
    "--instructions",
    default="",
    help="Free-form domain, tone, terminology, or product guidance",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path",
)
@_observability_options
def translate(
    model: str,
    key: str | None,
    host: str | None,
    temperature: float,
    po_file: str | None,
    po_dir: str | None,
    xliff_file: str | None,
    xliff_dir: str | None,
    source: str,
    lang: str | None,
    verbose: bool,
    context_file: str | None,
    max_page_units: int,
    max_page_chars: int,
    scope: str,
    jobs: int,
    order: str,
    profile: str,
    instructions: str,
    output: str | None,
    logfire: bool,
    logfire_capture_http: bool,
    mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str | None,
) -> None:
    """Translate PO/XLIFF files."""
    sources = [po_file, po_dir, xliff_file, xliff_dir]
    if not any(sources):
        click.echo(
            "Error: one of --po, --po-dir, --xliff, --xliff-dir is required", err=True
        )
        sys.exit(1)

    active = [s for s in sources if s]
    if len(active) > 1:
        click.echo(
            "Error: --po, --po-dir, --xliff, --xliff-dir are mutually exclusive",
            err=True,
        )
        sys.exit(1)

    with (
        _observability(
            logfire=logfire,
            logfire_capture_http=logfire_capture_http,
            mlflow=mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        ),
        _progress_reporter() as report_progress,
    ):
        kwargs = {
            "model": model,
            "source_lang": source,
            "target_lang": lang or "",
            "context_file": context_file,
            "max_page_units": max_page_units,
            "max_page_chars": max_page_chars,
            "scope": scope,
            "api_key": key,
            "api_host": host,
            "temperature": temperature,
            "on_progress": report_progress,
            "instructions": instructions,
        }

        if po_file:
            artifact = translate_file(
                input_path=po_file,
                output_path=output or po_file,
                order=order,
                profile=profile,
                **kwargs,
            )
        elif po_dir:
            artifact = translate_directory(
                dir_path=po_dir,
                suffixes=(".po",),
                jobs=jobs,
                order=order,
                profile=profile,
                **kwargs,
            )
        elif xliff_file:
            artifact = translate_file(
                input_path=xliff_file,
                output_path=output or xliff_file,
                order=order,
                profile=profile,
                **kwargs,
            )
        elif xliff_dir:
            artifact = translate_directory(
                dir_path=xliff_dir,
                suffixes=(".xliff", ".xlf"),
                jobs=jobs,
                order=order,
                profile=profile,
                **kwargs,
            )
        else:
            click.echo("No actionable target specified.", err=True)
            sys.exit(1)
        if verbose:
            click.echo(json.dumps(artifact, ensure_ascii=False, indent=2))


@app.command(
    "review",
    context_settings=CONTEXT_SETTINGS,
    help="Review translated PO/XLIFF files using QA + LLM.",
)
@click.option(
    "-m",
    "--model",
    envvar=["AITRAN_REVIEWER_MODEL", "AITRAN_MODEL"],
    default="deepseek:deepseek-v4-flash",
    help=(
        "Model in <provider>:<model> format "
        "(e.g. openai:gpt-5.4-mini, anthropic:claude-haiku-4-5)"
    ),
)
@click.option(
    "-k",
    "--key",
    envvar=["AITRAN_REVIEWER_KEY", "AITRAN_API_KEY"],
    help="API key for the LLM provider",
)
@click.option(
    "--host",
    envvar=["AITRAN_REVIEWER_HOST", "AITRAN_API_HOST"],
    help="Custom API base URL",
)
@click.option(
    "-t",
    "--temperature",
    envvar=["AITRAN_REVIEWER_TMP", "AITRAN_MODEL_TMP"],
    type=float,
    default=0.1,
    help="LLM temperature (0.0-2.0)",
)
@click.option("--po", "po_file", type=click.Path(exists=True), help="PO file path")
@click.option(
    "--xliff",
    "xliff_file",
    type=click.Path(exists=True),
    help="XLIFF/XLF file path",
)
@click.option("-src", "--source", default="en", help="Source language (ISO 639-1)")
@click.option("-l", "--lang", help="Target language (ISO 639-1)")
@click.option(
    "--max-page-units",
    "--batch-size",
    "max_page_units",
    type=click.IntRange(min=1),
    default=100,
    help="Max units per reviewer retrieval page",
)
@click.option(
    "--max-page-chars",
    type=click.IntRange(min=1),
    default=4096,
    show_default=True,
    help="Max source/target characters per reviewer retrieval page",
)
@click.option(
    "--instructions",
    default="",
    help="Free-form domain, tone, terminology, or product review guidance",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path (default: overwrite input)",
)
@_observability_options
def review(
    model: str,
    key: str | None,
    host: str | None,
    temperature: float,
    po_file: str | None,
    xliff_file: str | None,
    source: str,
    lang: str | None,
    max_page_units: int,
    max_page_chars: int,
    instructions: str,
    output: str | None,
    logfire: bool,
    logfire_capture_http: bool,
    mlflow: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment: str | None,
) -> None:
    """Review translated PO/XLIFF files using QA + LLM.

    Runs rule-based QA checks, then sends problematic units to an LLM
    reviewer for final verdict (pass/revise/reject).
    """
    if not po_file and not xliff_file:
        click.echo("Error: --po or --xliff is required", err=True)
        sys.exit(1)
    if po_file and xliff_file:
        click.echo("Error: --po and --xliff are mutually exclusive", err=True)
        sys.exit(1)

    with (
        _observability(
            logfire=logfire,
            logfire_capture_http=logfire_capture_http,
            mlflow=mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
        ),
        _progress_reporter() as report_progress,
    ):
        review_path: str = po_file or xliff_file  # guaranteed by mutual-exclusion check
        artifact = review_file(
            model=model,
            path=review_path,
            source_lang=source,
            target_lang=lang or "",
            output_path=output or review_path,
            max_page_units=max_page_units,
            max_page_chars=max_page_chars,
            api_key=key,
            api_host=host,
            temperature=temperature,
            on_progress=report_progress,
            instructions=instructions,
        )

    summary = artifact.get("summary") or {}
    total = sum(summary.values())
    click.echo(
        f"\nReviewed: {total} units\n"
        f"  pass:   {summary.get('pass', 0)}\n"
        f"  revise: {summary.get('revise', 0)}\n"
        f"  reject: {summary.get('reject', 0)}\n"
        f"Output: {artifact['output_path']}"
    )


@app.command("sync", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--po",
    "po_path",
    required=True,
    type=click.Path(exists=True),
    help="PO file path",
)
@click.option(
    "--pot",
    "pot_path",
    required=True,
    type=click.Path(exists=True),
    help="POT file path",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path",
)
def sync_cmd(po_path: str, pot_path: str, output: str | None) -> None:
    """Update PO file from a POT file, preserving existing translations."""
    sync(po_path, pot_path, output or po_path)
    click.echo("Sync complete.")


@app.command("remove", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--po",
    "po_path",
    required=True,
    type=click.Path(exists=True),
    help="PO file path",
)
@click.option("--fuzzy", is_flag=True, help="Remove fuzzy entries")
@click.option("-obs", "--obsolete", is_flag=True, help="Remove obsolete entries")
@click.option("-ut", "--untranslated", is_flag=True, help="Remove untranslated entries")
@click.option("-t", "--translated", is_flag=True, help="Remove translated entries")
@click.option(
    "-tnf",
    "--translated-not-fuzzy",
    is_flag=True,
    help="Remove translated non-fuzzy entries",
)
@click.option(
    "-ft",
    "--fuzzy-translated",
    is_flag=True,
    help="Remove fuzzy translated entries",
)
@click.option(
    "-rc",
    "--reference-contains",
    help="Remove entries whose reference matches text or /regex/flags",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path",
)
def remove(
    po_path: str,
    fuzzy: bool,
    obsolete: bool,
    untranslated: bool,
    translated: bool,
    translated_not_fuzzy: bool,
    fuzzy_translated: bool,
    reference_contains: str | None,
    output: str | None,
) -> None:
    """Remove PO entries matching filter criteria."""
    remove_by_options(
        po_path=po_path,
        output=output or po_path,
        fuzzy=fuzzy,
        obsolete=obsolete,
        untranslated=untranslated,
        translated=translated,
        translated_not_fuzzy=translated_not_fuzzy,
        fuzzy_translated=fuzzy_translated,
        reference_contains=reference_contains,
    )
    click.echo("Done.")


@app.group("weblate", context_settings=CONTEXT_SETTINGS)
def weblate() -> None:
    """Download or upload translation files using Weblate."""


@weblate.command("ls", context_settings=CONTEXT_SETTINGS, help="List Weblate objects.")
@_weblate_auth_options
@click.argument("object_path", required=False)
def weblate_ls(url: str, token: str, object_path: str | None) -> None:
    """List Weblate projects or child objects.

    Raises:
        click.ClickException: If listing fails.
    """
    try:
        click.echo(weblate_list_objects(url=url, token=token, object_path=object_path))
    except (TypeError, ValueError, WeblateException, RequestException) as exc:
        raise click.ClickException(str(exc)) from exc


@weblate.command(
    "stats",
    context_settings=CONTEXT_SETTINGS,
    help="Show Weblate object statistics.",
)
@_weblate_auth_options
@click.argument("object_path")
def weblate_stats(url: str, token: str, object_path: str) -> None:
    """Show Weblate statistics for a project, component, or translation.

    Raises:
        click.ClickException: If loading stats fails.
    """
    try:
        click.echo(weblate_get_stats(url=url, token=token, object_path=object_path))
    except (TypeError, ValueError, WeblateException, RequestException) as exc:
        raise click.ClickException(str(exc)) from exc


@weblate.command(
    "download",
    context_settings=CONTEXT_SETTINGS,
    help="Download a translation file from Weblate.",
)
@_weblate_auth_options
@click.option(
    "--object",
    "object_path",
    required=True,
    help="Weblate object path (<project>/<component>/<language>)",
)
@click.option(
    "-f",
    "--format",
    "download_format",
    type=click.Choice(WEBLATE_DOWNLOAD_FORMATS),
    help="Download format (defaults to output extension)",
)
@click.option(
    "--untranslated-only",
    is_flag=True,
    help="Download only untranslated strings",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output file path",
)
def weblate_download(
    url: str,
    token: str,
    object_path: str,
    download_format: str | None,
    untranslated_only: bool,
    output_path: str,
) -> None:
    """Download a translation file from Weblate.

    Raises:
        click.ClickException: If the download fails.
    """
    try:
        weblate_download_translation(
            url=url,
            token=token,
            object_path=object_path,
            output_path=output_path,
            download_format=download_format,
            untranslated_only=untranslated_only,
        )
    except (TypeError, ValueError, WeblateException, RequestException) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Download complete.")


@weblate.command(
    "upload",
    context_settings=CONTEXT_SETTINGS,
    help="Upload a translation file to Weblate.",
)
@_weblate_auth_options
@click.option(
    "--object",
    "object_path",
    required=True,
    help="Weblate object path (<project>/<component>/<language>)",
)
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Translation file to upload",
)
@click.option(
    "--method",
    type=click.Choice(WEBLATE_UPLOAD_METHODS),
    default="translate",
    show_default=True,
    help="Upload method",
)
@click.option(
    "--fuzzy",
    type=click.Choice(WEBLATE_FUZZY_CHOICES),
    help="Fuzzy string handling",
)
@click.option(
    "--conflicts",
    type=click.Choice(WEBLATE_CONFLICT_CHOICES),
    default="ignore",
    show_default=True,
    help="How to handle strings already translated in Weblate",
)
def weblate_upload(
    url: str,
    token: str,
    object_path: str,
    file_path: str,
    method: str,
    fuzzy: str | None,
    conflicts: str,
) -> None:
    """Upload a translation file to Weblate.

    Raises:
        click.ClickException: If the upload fails.
    """
    try:
        result = weblate_upload_translation(
            url=url,
            token=token,
            object_path=object_path,
            file_path=file_path,
            method=method,
            fuzzy=fuzzy,
            conflicts=conflicts,
        )
    except (TypeError, ValueError, WeblateException, RequestException) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))


@app.group("crowdin", context_settings=CONTEXT_SETTINGS)
def crowdin() -> None:
    """Download or upload translation files using Crowdin."""


@crowdin.command(
    "projects",
    context_settings=CONTEXT_SETTINGS,
    help="List Crowdin projects.",
)
@_crowdin_auth_options
@_timeout_option
def crowdin_projects(
    token: str,
    organization: str | None,
    base_url: str | None,
    timeout_seconds: int,
) -> None:
    """List Crowdin projects.

    Raises:
        click.ClickException: If listing fails.
    """
    try:
        click.echo(
            crowdin_list_projects(
                token=token,
                organization=organization,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        )
    except (CrowdinException, RequestException, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@crowdin.command("files", context_settings=CONTEXT_SETTINGS, help="List Crowdin files.")
@_crowdin_auth_options
@_crowdin_project_options
@_timeout_option
def crowdin_files(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int | None,
    project_name: str | None,
    timeout_seconds: int,
) -> None:
    """List Crowdin source files.

    Raises:
        click.ClickException: If listing fails.
    """
    try:
        click.echo(
            crowdin_list_files(
                token=token,
                organization=organization,
                base_url=base_url,
                project_id=project_id,
                project=project_name,
                timeout_seconds=timeout_seconds,
            )
        )
    except (CrowdinException, RequestException, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@crowdin.command(
    "languages",
    context_settings=CONTEXT_SETTINGS,
    help="List Crowdin supported languages.",
)
@_crowdin_auth_options
@_crowdin_project_options
@_timeout_option
def crowdin_languages(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int | None,
    project_name: str | None,
    timeout_seconds: int,
) -> None:
    """List Crowdin supported languages.

    Raises:
        click.ClickException: If listing fails.
    """
    try:
        click.echo(
            crowdin_list_languages(
                token=token,
                organization=organization,
                base_url=base_url,
                project_id=project_id,
                project=project_name,
                timeout_seconds=timeout_seconds,
            )
        )
    except (CrowdinException, RequestException, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@crowdin.command(
    "progress",
    context_settings=CONTEXT_SETTINGS,
    help="Show Crowdin translation progress.",
)
@_crowdin_auth_options
@_crowdin_project_options
@click.option("--file-id", type=int, help="Crowdin file ID for file progress")
@click.option("-l", "--lang", "language", help="Language ID for language progress")
@_timeout_option
def crowdin_progress(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int | None,
    project_name: str | None,
    file_id: int | None,
    language: str | None,
    timeout_seconds: int,
) -> None:
    """Show Crowdin project, file, or language progress.

    Raises:
        click.ClickException: If loading progress fails.
    """
    try:
        click.echo(
            crowdin_get_progress(
                token=token,
                organization=organization,
                base_url=base_url,
                project_id=project_id,
                project=project_name,
                file_id=file_id,
                language=language,
                timeout_seconds=timeout_seconds,
            )
        )
    except (CrowdinException, RequestException, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@crowdin.command(
    "download",
    context_settings=CONTEXT_SETTINGS,
    help="Download a translation file from Crowdin.",
)
@_crowdin_auth_options
@_crowdin_project_options
@click.option("--file-id", type=int, help="Crowdin file ID")
@click.option("-l", "--lang", "language", required=True, help="Target language code")
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output file path",
)
@_timeout_option
def crowdin_download(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int | None,
    project_name: str | None,
    file_id: int | None,
    language: str,
    output_path: str,
    timeout_seconds: int,
) -> None:
    """Download a translation file from Crowdin.

    Raises:
        click.ClickException: If the download fails.
    """
    try:
        crowdin_download_translation(
            token=token,
            organization=organization,
            base_url=base_url,
            project_id=project_id,
            project=project_name,
            file_id=file_id,
            language=language,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
        )
    except (CrowdinException, RequestException, TimeoutError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Download complete.")


@crowdin.command(
    "upload",
    context_settings=CONTEXT_SETTINGS,
    help="Upload a translation file to Crowdin.",
)
@_crowdin_auth_options
@_crowdin_project_options
@click.option("--file-id", type=int, help="Crowdin file ID")
@click.option("-l", "--lang", "language", required=True, help="Target language code")
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Translation file to upload",
)
@_timeout_option
def crowdin_upload(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int | None,
    project_name: str | None,
    file_id: int | None,
    language: str,
    file_path: str,
    timeout_seconds: int,
) -> None:
    """Upload a translation file to Crowdin.

    Raises:
        click.ClickException: If the upload fails.
    """
    try:
        crowdin_upload_translation(
            token=token,
            organization=organization,
            base_url=base_url,
            project_id=project_id,
            project=project_name,
            file_id=file_id,
            language=language,
            file_path=file_path,
            timeout_seconds=timeout_seconds,
        )
    except (CrowdinException, RequestException, TimeoutError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Upload complete.")


@app.command("userdict", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--explore",
    is_flag=True,
    help="Open dictionary directory in file manager",
)
@click.option("-l", "--lang", help="Target language (ISO 639-1)")
def userdict(explore: bool, lang: str | None) -> None:
    """Open or explore user dictionaries."""
    default_dict = str(files("aitran").parent.parent / "dictionary.json")

    dict_filename = f"dictionary{'-' + lang if lang else ''}.json"
    dict_file = find_config(dict_filename)

    if explore:
        open_file_explorer(dict_file)
    else:
        if not lang:
            copy_file_if_not_exists(dict_file, default_dict)
        open_file_by_default(dict_file)
