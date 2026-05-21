"""CLI entry point using click."""

import os
import sys

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


def _ensure_api_key() -> None:
    """Check that an API key is set (AITRAN_API_KEY or OPENAI_API_KEY)."""
    if not os.environ.get("AITRAN_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        click.echo(
            "Error: API key is required. Set AITRAN_API_KEY or OPENAI_API_KEY.",
            err=True,
        )
        sys.exit(1)


@click.group(invoke_without_command=True)
@click.version_option(message="%(prog)s %(version)s")
@click.pass_context
def app(ctx: click.Context) -> None:
    """aitran — Translate PO and XLIFF files using LLMs.

    Supports OpenAI, Anthropic, and 100+ providers via litellm.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(translate)


def _get_compile_options(
    po_fold_len: str, po_sort: bool, po_esc_chars: bool
) -> dict:
    """Parse PO compile options from CLI args."""
    try:
        fold_len = 0 if po_fold_len.lower() == "false" else int(po_fold_len)
    except ValueError:
        click.echo("--po-fold-len must be a number or 'false'", err=True)
        sys.exit(1)
    return {"fold_length": fold_len, "sort_output": po_sort, "escape_chars": po_esc_chars}


# Shared options for PO compile parameters
_po_fold_len = click.option(
    "--po-fold-len",
    default="120",
    help="Fold length for PO output; 0 or 'false' disables folding",
)
_po_sort = click.option(
    "--po-sort", is_flag=True, default=False, help="Sort PO entries by msgid"
)
_po_esc_chars = click.option(
    "--po-esc-chars/--no-po-esc-chars",
    default=True,
    help="Escape characters in PO output",
)


@app.command()
@click.option(
    "-m", "--model",
    envvar="AITRAN_MODEL",
    default="gpt-4o-mini",
    help="Model name in litellm format (e.g. gpt-4o, claude-sonnet-4-20250514)",
)
@click.option(
    "-k", "--key", envvar="AITRAN_API_KEY", help="API key for the LLM provider"
)
@click.option("--host", envvar="AITRAN_API_HOST", help="Custom API base URL")
@click.option("--po", "po_file", type=click.Path(exists=True), help="PO file path")
@click.option(
    "--dir", "po_dir", type=click.Path(exists=True, file_okay=False),
    help="Directory of .po files",
)
@click.option(
    "--xliff", "xliff_file", type=click.Path(exists=True),
    help="XLIFF file path",
)
@click.option(
    "--xliff-dir", type=click.Path(exists=True, file_okay=False),
    help="Directory of .xliff/.xlf files",
)
@click.option(
    "-src", "--source", default="en", help="Source language (ISO 639-1)"
)
@click.option("-l", "--lang", help="Target language (ISO 639-1)")
@click.option("-v", "--verbose", is_flag=True, help="Print each translation")
@click.option(
    "--context", "context_file", type=click.Path(exists=True),
    help="Text file with additional translation context",
)
@click.option(
    "--context-length", type=int, default=2000,
    help="Max accumulated source length per API batch",
)
@click.option(
    "--timeout", type=int, default=20000, help="API timeout in milliseconds",
)
@click.option(
    "-o", "--output", type=click.Path(), help="Output file path",
)
@_po_fold_len
@_po_sort
@_po_esc_chars
def translate(
    model: str,
    key: str | None,
    host: str | None,
    po_file: str | None,
    po_dir: str | None,
    xliff_file: str | None,
    xliff_dir: str | None,
    source: str,
    lang: str | None,
    verbose: bool,
    context_file: str | None,
    context_length: int,
    timeout: int,
    output: str | None,
    po_fold_len: str,
    po_sort: bool,
    po_esc_chars: bool,
) -> None:
    """Translate PO/XLIFF files (default command)."""
    if key:
        os.environ["AITRAN_API_KEY"] = key
    if host:
        os.environ["AITRAN_API_HOST"] = host

    _ensure_api_key()

    if not any([po_file, po_dir, xliff_file, xliff_dir]):
        click.echo(
            "Error: one of --po, --dir, --xliff, --xliff-dir is required", err=True
        )
        sys.exit(1)

    # Count how many sources are specified (mutual exclusion check)
    sources = [po_file, po_dir, xliff_file, xliff_dir]
    active = [s for s in sources if s]
    if len(active) > 1:
        click.echo(
            "Error: --po, --dir, --xliff, --xliff-dir are mutually exclusive",
            err=True,
        )
        sys.exit(1)

    compile_opts = _get_compile_options(po_fold_len, po_sort, po_esc_chars)

    if po_file:
        translate_po(
            model=model,
            po_path=po_file,
            source_lang=source,
            target_lang=lang or "",
            verbose=verbose,
            output_path=output or po_file,
            context_file=context_file,
            context_length=context_length,
            timeout=timeout,
            **compile_opts,
        )
    elif po_dir:
        translate_po_dir(
            model=model,
            dir_path=po_dir,
            source_lang=source,
            target_lang=lang or "",
            verbose=verbose,
            context_file=context_file,
            context_length=context_length,
            timeout=timeout,
            **compile_opts,
        )
    elif xliff_file:
        translate_xliff_file(
            model=model,
            xliff_path=xliff_file,
            source_lang=source,
            target_lang=lang or "",
            verbose=verbose,
            output_path=output or xliff_file,
            context_file=context_file,
            context_length=context_length,
            timeout=timeout,
        )
    elif xliff_dir:
        translate_xliff_dir(
            model=model,
            dir_path=xliff_dir,
            source_lang=source,
            target_lang=lang or "",
            verbose=verbose,
            context_file=context_file,
            context_length=context_length,
            timeout=timeout,
        )
    else:
        click.echo("No actionable target specified.", err=True)
        sys.exit(1)


@app.command("sync")
@click.option(
    "--po", "po_path", required=True, type=click.Path(exists=True),
    help="PO file path",
)
@click.option(
    "--pot", "pot_path", required=True, type=click.Path(exists=True),
    help="POT file path",
)
@click.option(
    "-o", "--output", type=click.Path(), help="Output file path",
)
@_po_fold_len
@_po_sort
@_po_esc_chars
def sync_cmd(
    po_path: str,
    pot_path: str,
    output: str | None,
    po_fold_len: str,
    po_sort: bool,
    po_esc_chars: bool,
) -> None:
    """Update PO file from a POT file, preserving existing translations."""
    compile_opts = _get_compile_options(po_fold_len, po_sort, po_esc_chars)
    sync(po_path, pot_path, output or po_path, **compile_opts)
    click.echo("Sync complete.")


@app.command()
@click.option(
    "--po", "po_path", required=True, type=click.Path(exists=True),
    help="PO file path",
)
@click.option("--fuzzy", is_flag=True, help="Remove fuzzy entries")
@click.option(
    "-obs", "--obsolete", is_flag=True, help="Remove obsolete entries"
)
@click.option(
    "-ut", "--untranslated", is_flag=True, help="Remove untranslated entries"
)
@click.option(
    "-t", "--translated", is_flag=True, help="Remove translated entries"
)
@click.option(
    "-tnf", "--translated-not-fuzzy", is_flag=True,
    help="Remove translated non-fuzzy entries",
)
@click.option(
    "-ft", "--fuzzy-translated", is_flag=True,
    help="Remove fuzzy translated entries",
)
@click.option(
    "-rc", "--reference-contains",
    help="Remove entries whose reference matches text or /regex/flags",
)
@click.option(
    "-o", "--output", type=click.Path(), help="Output file path",
)
@_po_fold_len
@_po_sort
@_po_esc_chars
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
    po_fold_len: str,
    po_sort: bool,
    po_esc_chars: bool,
) -> None:
    """Remove PO entries matching filter criteria."""
    compile_opts = _get_compile_options(po_fold_len, po_sort, po_esc_chars)
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
        compile_opts=compile_opts,
    )
    click.echo("Done.")


@app.command()
@click.option(
    "--explore", is_flag=True, help="Open dictionary directory in file manager",
)
@click.option("-l", "--lang", help="Target language (ISO 639-1)")
def userdict(explore: bool, lang: str | None) -> None:
    """Open or explore user dictionaries."""
    from importlib.resources import files

    home = os.path.expanduser("~")
    default_dict = str(files("aitran").parent.parent / "dictionary.json")

    dict_filename = f"dictionary{'-' + lang if lang else ''}.json"
    dict_file = find_config(dict_filename)

    if explore:
        open_file_explorer(dict_file)
    else:
        if not lang:
            copy_file_if_not_exists(dict_file, default_dict)
        open_file_by_default(dict_file)
