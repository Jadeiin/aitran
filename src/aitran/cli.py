"""CLI entry point using click."""

import sys
from contextlib import contextmanager
from importlib.resources import files

import click
from crowdin_api.exceptions import CrowdinException
from requests import RequestException
from wlc.client import WeblateException

from aitran.crowdin import download_translation as crowdin_download_translation
from aitran.crowdin import get_progress as crowdin_get_progress
from aitran.crowdin import list_files as crowdin_list_files
from aitran.crowdin import list_languages as crowdin_list_languages
from aitran.crowdin import list_projects as crowdin_list_projects
from aitran.crowdin import upload_translation as crowdin_upload_translation
from aitran.manipulate import remove_by_options
from aitran.observability import (
    ObservabilityError,
    flush_logfire,
    flush_mlflow,
    setup_logfire,
    setup_mlflow,
)
from aitran.review import review_file
from aitran.sync import sync
from aitran.translate import (
    translate_po,
    translate_po_dir,
    translate_xliff_dir,
    translate_xliff_file,
)
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
WEBLATE_DOWNLOAD_FORMATS = ["po", "xliff11", "xliff"]


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
    "-p",
    "--prompt",
    help="Initial natural-language request for the interactive app.",
)
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
    envvar="AITRAN_MODEL",
    default="deepseek:deepseek-v4-flash",
    show_default=True,
    help="Model for translation/review tasks",
)
@click.option(
    "--translate-key",
    envvar="AITRAN_API_KEY",
    help="API key for the translation model",
)
@click.option(
    "--translate-host",
    envvar="AITRAN_API_HOST",
    help="Custom API base URL for the translation model",
)
@click.option(
    "--translate-temperature",
    envvar="AITRAN_MODEL_TMP",
    type=float,
    default=0.1,
    show_default=True,
    help="LLM temperature for translation/review tasks.",
)
@click.option(
    "--session-id",
    help="Session ID to resume or name a new session",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from a saved session",
)
@click.option(
    "--auto-approve",
    is_flag=True,
    envvar="AITRAN_APP_AUTO_APPROVE",
    help="Automatically approve tools that require confirmation.",
)
@_observability_options
@click.pass_context
def app(
    ctx: click.Context,
    prompt: str | None,
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
    session_id: str | None,
    resume: bool,
    auto_approve: bool,
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

    from rich.console import Console

    from aitran.app import run_app
    from aitran.toolsets._base import OrchestratorDeps

    console = Console()

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
        )
        run_app(
            prompt,
            orchestrator_model=orchestrator_model,
            orchestrator_api_key=orchestrator_key,
            orchestrator_api_host=orchestrator_host,
            orchestrator_temperature=orchestrator_temperature,
            deps=deps,
            session_id=session_id,
            resume=resume,
            auto_approve=auto_approve,
            console=console,
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
@click.option("-v", "--verbose", is_flag=True, help="Print each translation")
@click.option(
    "--context",
    "context_file",
    type=click.Path(exists=True),
    help="Text file with additional translation context",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Max units per API batch",
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
        "Prompt detail level: fast (index + source only), "
        "full (all metadata: context, location, note, flag)"
    ),
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
    batch_size: int,
    jobs: int,
    order: str,
    profile: str,
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

    with _observability(
        logfire=logfire,
        logfire_capture_http=logfire_capture_http,
        mlflow=mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    ):
        kwargs = {
            "model": model,
            "source_lang": source,
            "target_lang": lang or "",
            "verbose": verbose,
            "context_file": context_file,
            "batch_size": batch_size,
            "api_key": key,
            "api_host": host,
            "temperature": temperature,
        }

        if po_file:
            translate_po(
                po_path=po_file,
                output_path=output or po_file,
                order=order,
                profile=profile,
                **kwargs,
            )
        elif po_dir:
            translate_po_dir(
                dir_path=po_dir,
                jobs=jobs,
                order=order,
                profile=profile,
                **kwargs,
            )
        elif xliff_file:
            translate_xliff_file(
                xliff_path=xliff_file,
                output_path=output or xliff_file,
                order=order,
                profile=profile,
                **kwargs,
            )
        elif xliff_dir:
            translate_xliff_dir(
                dir_path=xliff_dir,
                jobs=jobs,
                order=order,
                profile=profile,
                **kwargs,
            )
        else:
            click.echo("No actionable target specified.", err=True)
            sys.exit(1)


@app.command(
    "review",
    context_settings=CONTEXT_SETTINGS,
    help="Review translated PO/XLIFF files using QA + LLM.",
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
    "--xliff",
    "xliff_file",
    type=click.Path(exists=True),
    help="XLIFF/XLF file path",
)
@click.option("-src", "--source", default="en", help="Source language (ISO 639-1)")
@click.option("-l", "--lang", help="Target language (ISO 639-1)")
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Max units per review batch",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Review all units (default: only units with QA errors or markers)",
)
@click.option(
    "--auto-fix",
    is_flag=True,
    help="Write corrected targets back to the file",
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
    batch_size: int,
    strict: bool,
    auto_fix: bool,
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

    with _observability(
        logfire=logfire,
        logfire_capture_http=logfire_capture_http,
        mlflow=mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    ):
        review_path: str = po_file or xliff_file  # guaranteed by mutual-exclusion check
        summary = review_file(
            model=model,
            path=review_path,
            source_lang=source,
            target_lang=lang or "",
            output_path=output or review_path,
            batch_size=batch_size,
            strict=strict,
            auto_fix=auto_fix,
            api_key=key,
            api_host=host,
            temperature=temperature,
        )

    total = sum(summary.values())
    click.echo(
        f"\nReviewed: {total} units\n"
        f"  pass:   {summary.get('pass', 0)}\n"
        f"  revise: {summary.get('revise', 0)}\n"
        f"  reject: {summary.get('reject', 0)}\n"
        f"  skip:   {summary.get('skip', 0)}"
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
def weblate_upload(
    url: str,
    token: str,
    object_path: str,
    file_path: str,
    method: str,
    fuzzy: str | None,
) -> None:
    """Upload a translation file to Weblate.

    Raises:
        click.ClickException: If the upload fails.
    """
    try:
        weblate_upload_translation(
            url=url,
            token=token,
            object_path=object_path,
            file_path=file_path,
            method=method,
            fuzzy=fuzzy,
        )
    except (TypeError, ValueError, WeblateException, RequestException) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Upload complete.")


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
