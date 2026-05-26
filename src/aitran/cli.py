"""CLI entry point using click."""

import sys
from importlib.resources import files

import click
from crowdin_api.api_resources.enums import ExportProjectTranslationFormat
from crowdin_api.exceptions import CrowdinException
from requests import RequestException
from wlc.client import WeblateException

from aitran.crowdin import download_translation as crowdin_download_translation
from aitran.crowdin import upload_translation as crowdin_upload_translation
from aitran.manipulate import remove_by_options
from aitran.observability import ObservabilityError, flush_logfire, setup_logfire
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


def _parse_weblate_object(value: str) -> tuple[str, str, str]:
    parts = [part for part in value.strip("/").split("/") if part]
    if len(parts) != 3:
        raise click.ClickException(
            "Weblate object must be in '<project>/<component>/<language>' format."
        )
    return parts[0], parts[1], parts[2]


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(message="%(prog)s %(version)s")
def app() -> None:
    """Aitran — Translate PO and XLIFF files using LLMs.

    Built on Pydantic AI. Supports OpenAI, Anthropic, and any OpenAI-compatible
    provider; specify models as `<provider>:<model>` (e.g. `openai:gpt-5.4-mini`,
    `anthropic:claude-haiku-4-5`).
    """


@app.command(context_settings=CONTEXT_SETTINGS)
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
    "--context-length",
    type=int,
    default=4096,
    help="Max accumulated source length per API batch",
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
@click.option(
    "--logfire",
    is_flag=True,
    envvar="AITRAN_LOGFIRE",
    help=(
        "Enable Pydantic Logfire tracing for agent/model runs. "
        "Prompts and completions may be sent to Logfire."
    ),
)
@click.option(
    "--logfire-capture-http",
    is_flag=True,
    envvar="AITRAN_LOGFIRE_CAPTURE_HTTP",
    help=(
        "Also capture provider HTTP headers and bodies in Logfire. "
        "This may include prompts, completions, and credentials."
    ),
)
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
    context_length: int,
    jobs: int,
    order: str,
    profile: str,
    output: str | None,
    logfire: bool,
    logfire_capture_http: bool,
) -> None:
    """Translate PO/XLIFF files (default command).

    Raises:
        click.ClickException: If optional observability setup fails.
    """
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

    try:
        logfire_enabled = setup_logfire(
            enabled=logfire,
            capture_http=logfire_capture_http,
        )
    except ObservabilityError as exc:
        raise click.ClickException(str(exc)) from exc

    kwargs = {
        "model": model,
        "source_lang": source,
        "target_lang": lang or "",
        "verbose": verbose,
        "context_file": context_file,
        "context_length": context_length,
        "api_key": key,
        "api_host": host,
        "temperature": temperature,
    }

    try:
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
    finally:
        flush_logfire(enabled=logfire_enabled)


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


@weblate.command("download", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--url",
    envvar="AITRAN_WEBLATE_URL",
    required=True,
    help="Weblate base URL (e.g. https://weblate.example.org)",
)
@click.option(
    "--token",
    envvar="AITRAN_WEBLATE_TOKEN",
    required=True,
    help="Weblate API token",
)
@click.option(
    "--object",
    "object_path",
    required=True,
    help="Weblate object path (<project>/<component>/<language>)",
)
@click.option(
    "-c",
    "--convert",
    help="Convert file format on server (defaults to none)",
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
    convert: str | None,
    output_path: str,
) -> None:
    """Download a translation file from Weblate.

    Raises:
        click.ClickException: If the download fails.
    """
    try:
        project, component, language = _parse_weblate_object(object_path)
        weblate_download_translation(
            url=url,
            token=token,
            project=project,
            component=component,
            language=language,
            output_path=output_path,
            convert=convert,
        )
    except (ValueError, WeblateException) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Download complete.")


@weblate.command("upload", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--url",
    envvar="AITRAN_WEBLATE_URL",
    required=True,
    help="Weblate base URL (e.g. https://weblate.example.org)",
)
@click.option(
    "--token",
    envvar="AITRAN_WEBLATE_TOKEN",
    required=True,
    help="Weblate API token",
)
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
        project, component, language = _parse_weblate_object(object_path)
        weblate_upload_translation(
            url=url,
            token=token,
            project=project,
            component=component,
            language=language,
            file_path=file_path,
            method=method,
            fuzzy=fuzzy,
        )
    except (ValueError, WeblateException) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Upload complete.")


@app.group("crowdin", context_settings=CONTEXT_SETTINGS)
def crowdin() -> None:
    """Download or upload translation files using Crowdin."""


@crowdin.command("download", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--token",
    envvar="AITRAN_CROWDIN_TOKEN",
    required=True,
    help="Crowdin API token",
)
@click.option(
    "--organization",
    envvar="AITRAN_CROWDIN_ORG",
    help="Crowdin organization (Enterprise only)",
)
@click.option(
    "--base-url",
    envvar="AITRAN_CROWDIN_BASE_URL",
    help="Crowdin API base URL override",
)
@click.option("--project-id", type=int, required=True, help="Crowdin project ID")
@click.option("--file-id", type=int, required=True, help="Crowdin file ID")
@click.option("-l", "--lang", "language", required=True, help="Target language code")
@click.option(
    "--format",
    "export_format",
    type=click.Choice([value.value for value in ExportProjectTranslationFormat]),
    default=ExportProjectTranslationFormat.XLIFF.value,
    show_default=True,
    help="Export format",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output file path",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=click.IntRange(min=1),
    default=120,
    show_default=True,
    help="Timeout (seconds) for API operations",
)
@click.option(
    "--poll-interval",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Polling interval (seconds) for build completion",
)
def crowdin_download(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int,
    file_id: int,
    language: str,
    export_format: str,
    output_path: str,
    timeout_seconds: int,
    poll_interval: int,
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
            file_id=file_id,
            language=language,
            export_format=ExportProjectTranslationFormat(export_format),
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        )
    except (CrowdinException, RequestException, TimeoutError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Download complete.")


@crowdin.command("upload", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--token",
    envvar="AITRAN_CROWDIN_TOKEN",
    required=True,
    help="Crowdin API token",
)
@click.option(
    "--organization",
    envvar="AITRAN_CROWDIN_ORG",
    help="Crowdin organization (Enterprise only)",
)
@click.option(
    "--base-url",
    envvar="AITRAN_CROWDIN_BASE_URL",
    help="Crowdin API base URL override",
)
@click.option("--project-id", type=int, required=True, help="Crowdin project ID")
@click.option("--file-id", type=int, required=True, help="Crowdin file ID")
@click.option("-l", "--lang", "language", required=True, help="Target language code")
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Translation file to upload",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    type=click.IntRange(min=1),
    default=120,
    show_default=True,
    help="Timeout (seconds) for API operations",
)
def crowdin_upload(
    token: str,
    organization: str | None,
    base_url: str | None,
    project_id: int,
    file_id: int,
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
