"""CLI entry point using click."""

import sys
from importlib.resources import files

import click

from aitran.manipulate import remove_by_options
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

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


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
) -> None:
    """Translate PO/XLIFF files (default command)."""
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
