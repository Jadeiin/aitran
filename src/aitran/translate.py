"""Core translation engine built on a Pydantic AI agent."""

from __future__ import annotations

import asyncio
import html
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import TYPE_CHECKING

import lxml.etree as ET
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from translate.storage import po, xliff

from aitran.agent import (
    TranslationDeps,
    build_input_xml,
    build_model,
    build_translator_agent,
)
from aitran.dicts import find_matching_entries

if TYPE_CHECKING:
    from aitran.agent import TranslatedUnit


def _read_context(context_file: str | None) -> str:
    if not context_file:
        return ""
    with open(context_file, encoding="utf-8") as f:
        return f.read().strip()


def _is_rate_limit(exc: ModelHTTPError) -> bool:
    return exc.status_code == 429


def _is_timeout(exc: ModelHTTPError) -> bool:
    return exc.status_code in (408, 504)


def _build_progress(console: Console | None = None) -> Progress:
    """Create the shared progress renderer used by file translations.

    Returns:
        Rich progress renderer with file labels.
    """
    progress = Progress(
        TextColumn("[cyan]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console or Console(),
    )
    progress.live.vertical_overflow = "crop"
    return progress


class PoTranslator:
    """Handles PO file parsing, filtering, and output."""

    @staticmethod
    def parse(path: str) -> po.pofile:
        """Parse a PO file from disk.

        Returns:
            Parsed PO file object.
        """
        return po.pofile.parsefile(path)

    @staticmethod
    def get_header_language(po_file: po.pofile) -> str | None:
        """Extract the target language from the PO header.

        Returns:
            Target language string, or None if not set.
        """
        lang = po_file.gettargetlanguage()
        return lang if lang else None

    @staticmethod
    def get_untranslated(po_file: po.pofile) -> list[po.pounit]:
        """Return units that need translation (empty target or fuzzy).

        Returns:
            List of untranslated or fuzzy PO units.
        """
        result: list[po.pounit] = []
        for unit in po_file.units:
            if unit.isheader():
                continue
            if unit.istranslated() and not unit.isfuzzy():
                continue
            result.append(unit)
        return result

    @staticmethod
    def apply_batch(
        _po_file: po.pofile,
        units: list[po.pounit],
        results: list[TranslatedUnit],
    ) -> None:
        """Apply a batch of agent results."""
        for unit, result in zip(units, results, strict=True):
            unit.target = result.target
            unit.markfuzzy(result.fuzzy)
            if result.note:
                unit.addnote(result.note, origin="translator")

    @staticmethod
    def save(po_file: po.pofile, path: str) -> None:
        """Serialize a PO file to disk.

        Args:
            po_file: The in-memory PO file.
            path: Destination file path.
        """
        with open(path, "wb") as f:
            f.write(bytes(po_file))


class XliffTranslator:
    """Handles XLIFF file parsing, filtering, and output."""

    _XLIFF_NS = "{urn:oasis:names:tc:xliff:document:1.2}"

    @staticmethod
    def parse(path: str) -> xliff.xlifffile:
        """Parse an XLIFF file from disk.

        Returns:
            Parsed XLIFF file object.
        """
        return xliff.xlifffile.parsefile(path)

    @staticmethod
    def _get_state(unit: xliff.xliffunit) -> str:
        target_elem = unit.xmlelement.find(f"{XliffTranslator._XLIFF_NS}target")
        if target_elem is not None:
            return target_elem.get("state", "")
        return ""

    @staticmethod
    def _get_translate_flag(unit: xliff.xliffunit) -> bool:
        return unit.xmlelement.get("translate", "yes").lower() != "no"

    @classmethod
    def get_untranslated(cls, xlf: xliff.xlifffile) -> list[xliff.xliffunit]:
        """Return units that need translation."""
        result: list[xliff.xliffunit] = []
        for unit in xlf.units:
            if not cls._get_translate_flag(unit):
                continue
            state = cls._get_state(unit).lower()
            target = (unit.target or "").strip()
            source = (unit.source or "").strip()

            state_needs = state.startswith("needs-") or state in ("new", "")
            has_meaningful = bool(target) and target != source and not state_needs
            if not has_meaningful:
                result.append(unit)
        return result

    @staticmethod
    def apply_batch(
        _xlf: xliff.xlifffile,
        units: list[xliff.xliffunit],
        results: list[TranslatedUnit],
    ) -> None:
        """Apply translation results to XLIFF units."""
        for unit, result in zip(units, results, strict=True):
            unit.target = result.target
            target_elem = unit.xmlelement.find(f"{XliffTranslator._XLIFF_NS}target")
            new_state = "needs-review-translation" if result.fuzzy else "translated"
            if target_elem is not None:
                target_elem.set("state", new_state)
            elif result.target:
                new_target = ET.SubElement(
                    unit.xmlelement,
                    f"{XliffTranslator._XLIFF_NS}target",
                    {"state": new_state},
                )
                new_target.text = result.target
            if result.note:
                unit.addnote(result.note, origin="translator")

    @staticmethod
    def save(xlf: xliff.xlifffile, path: str) -> None:
        """Serialize an XLIFF file to disk.

        Args:
            xlf: The in-memory XLIFF file.
            path: Destination file path.
        """
        with open(path, "wb") as f:
            f.write(bytes(xlf))


async def _translate_batch(
    agent,
    units: list,
    start_index: int,
    deps: TranslationDeps,
    history: list,
    on_progress,
) -> list[TranslatedUnit]:
    """Stream one batch through the agent.

    Returns:
        List of TranslatedUnit aligned with the input units list.
    """
    user_msg = build_input_xml(units, start_index)
    seen: set[int] = set()

    async with agent.run_stream(
        user_msg,
        deps=deps,
        message_history=history,
    ) as run:
        async for partial in run.stream_output(debounce_by=0.1):
            for t in partial.translations:
                if t.index in seen:
                    continue
                local_idx = t.index - start_index
                if not (0 <= local_idx < len(units)):
                    continue
                if not t.target:
                    continue
                seen.add(t.index)
                if on_progress:
                    on_progress(units[local_idx].source, t)
        final = await run.get_output()
        history.extend(run.new_messages())

    by_index: dict[int, TranslatedUnit] = {t.index: t for t in final.translations}
    results = []
    for i in range(len(units)):
        tu = by_index[start_index + i]
        # Reverse XML escaping applied by format_as_xml so that raw HTML
        # tags (e.g. <code>) in the source map to unescaped tags in the target.
        tu.target = html.unescape(tu.target)
        results.append(tu)
    return results


async def _run_translation_async(
    store,  # pofile | xlifffile
    units: list,
    source_lang: str,
    target_lang: str,
    model_spec: str,
    translator,  # PoTranslator | XliffTranslator
    output_path: str,
    context_file: str | None,
    context_length: int,
    verbose: bool,
    progress_label: str,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> None:
    """Shared batch loop driving the translator agent."""
    context = _read_context(context_file)
    sources = [u.source for u in units]
    dict_entries = find_matching_entries(sources, target_lang)

    base_url = (api_host.rstrip("/") + "/v1") if api_host else None
    agent = build_translator_agent(
        build_model(
            model_spec, api_key=api_key, base_url=base_url, temperature=temperature
        )
    )
    history: list = []

    owns_progress = progress is None
    progress = progress or _build_progress()
    console = progress.console
    task_id = progress.add_task(progress_label, total=len(units))
    global_done = 0
    batch_streamed: set[int] = set()
    saved_positions: set[int] = set()

    def on_unit_done(src: str, result: TranslatedUnit) -> None:
        nonlocal global_done
        pos = result.index - 1  # 1-based → 0-based
        if pos in saved_positions or pos in batch_streamed:
            return
        batch_streamed.add(pos)
        global_done += 1
        progress.update(task_id, completed=global_done)
        if verbose:
            src_short = src[:70] + ("…" if len(src) > 70 else "")
            tgt_short = result.target[:60] + ("…" if len(result.target) > 60 else "")
            flag = " [yellow][fuzzy][/]" if result.fuzzy else ""
            progress.console.print(
                f"[cyan]{progress_label}[/] {src_short} → {tgt_short}{flag}"
            )

    def _commit_batch() -> None:
        """Mark the current batch's streamed positions as permanently saved."""
        nonlocal global_done
        saved_positions.update(batch_streamed)
        batch_streamed.clear()

    def _rollback_batch() -> None:
        """Undo progress from a failed batch attempt so the bar reflects reality."""
        nonlocal global_done
        global_done -= len(batch_streamed)
        batch_streamed.clear()
        progress.update(task_id, completed=global_done)

    batch: list = []
    char_count = 0
    i = 0
    next_start_index = 1
    err429 = False
    batch_retries = 0
    BATCH_MAX_RETRIES = 3

    with progress if owns_progress else nullcontext():
        while i < len(units):
            if err429:
                await asyncio.sleep(20)
                err429 = False

            unit = units[i]
            src_len = len(unit.source)
            if char_count < context_length:
                batch.append(unit)
                char_count += src_len
            if char_count >= context_length or i == len(units) - 1:
                deps = TranslationDeps(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context=context,
                    dict_entries=dict_entries,
                    expected_indices=tuple(
                        range(next_start_index, next_start_index + len(batch))
                    ),
                )
                try:
                    results = await _translate_batch(
                        agent,
                        batch,
                        next_start_index,
                        deps,
                        history,
                        on_unit_done,
                    )
                    translator.apply_batch(store, batch, results)
                    translator.save(store, output_path)
                    _commit_batch()
                    next_start_index += len(batch)
                    batch = []
                    char_count = 0
                    batch_retries = 0
                except ModelHTTPError as e:
                    if _is_rate_limit(e):
                        err429 = True
                        continue
                    if _is_timeout(e):
                        console.print("\n[yellow]Timeout. Retrying...[/]")
                        continue
                    console.print(f"\n[red]HTTP error {e.status_code}: {e}[/]")
                    continue
                except UnexpectedModelBehavior as e:
                    batch_retries += 1
                    cause = e.__cause__
                    cause_msg = f": {cause}" if cause is not None else ""
                    if batch_retries < BATCH_MAX_RETRIES:
                        _rollback_batch()
                        console.print(
                            f"\n[yellow]Output validation failed{cause_msg}. "
                            f"Retrying batch "
                            f"({batch_retries}/{BATCH_MAX_RETRIES})...[/]"
                        )
                        continue
                    console.print(
                        f"\n[red]Output validation failed after "
                        f"{BATCH_MAX_RETRIES} retries{cause_msg}. "
                        f"Skipping {len(batch)} unit(s).[/]"
                    )
                    _commit_batch()
                    next_start_index += len(batch)
                    batch = []
                    char_count = 0
                    batch_retries = 0

            i += 1


def _run_translation(
    store,
    units: list,
    source_lang: str,
    target_lang: str,
    model_spec: str,
    translator,
    output_path: str,
    context_file: str | None,
    context_length: int,
    verbose: bool,
    progress_label: str,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> None:
    asyncio.run(
        _run_translation_async(
            store=store,
            units=units,
            source_lang=source_lang,
            target_lang=target_lang,
            model_spec=model_spec,
            translator=translator,
            output_path=output_path,
            context_file=context_file,
            context_length=context_length,
            verbose=verbose,
            progress_label=progress_label,
            api_key=api_key,
            api_host=api_host,
            temperature=temperature,
            progress=progress,
        )
    )


def translate_po(
    model: str,
    po_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    output_path: str,
    context_file: str | None,
    context_length: int,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> None:
    """Translate a single PO file."""
    translator = PoTranslator()
    po_file = translator.parse(po_path)

    if not target_lang:
        header_lang = translator.get_header_language(po_file)
        if header_lang:
            target_lang = header_lang
    if not target_lang:
        print("No target language specified via --lang or PO header", file=sys.stderr)
        return

    untranslated = translator.get_untranslated(po_file)
    if not untranslated:
        print("All entries already translated.")
        translator.save(po_file, output_path)
        return

    po_file.updateheader(**{"Last-Translator": "aitran v0.1.0"})

    untranslated.sort(key=lambda u: u.source)

    _run_translation(
        store=po_file,
        units=untranslated,
        source_lang=source_lang,
        target_lang=target_lang,
        model_spec=model,
        translator=translator,
        output_path=output_path,
        context_file=context_file,
        context_length=context_length,
        verbose=verbose,
        progress_label=os.path.basename(output_path),
        api_key=api_key,
        api_host=api_host,
        temperature=temperature,
        progress=progress,
    )


def translate_po_dir(
    model: str,
    dir_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    context_file: str | None,
    context_length: int,
    jobs: int = 4,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
) -> None:
    """Translate all .po files in a directory."""
    po_paths = [
        os.path.join(dir_path, entry)
        for entry in sorted(os.listdir(dir_path))
        if entry.endswith(".po")
    ]
    if not po_paths:
        print("No .po files found.")
        return

    max_workers = min(jobs, len(po_paths))
    progress = _build_progress()
    with progress, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                translate_po,
                model,
                po_path,
                source_lang,
                target_lang,
                verbose,
                po_path,
                context_file,
                context_length,
                api_key=api_key,
                api_host=api_host,
                temperature=temperature,
                progress=progress,
            ): po_path
            for po_path in po_paths
        }
        for future in as_completed(futures):
            future.result()


def translate_xliff_file(
    model: str,
    xliff_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    output_path: str,
    context_file: str | None,
    context_length: int,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> None:
    """Translate a single XLIFF file."""
    translator = XliffTranslator()
    xlf = translator.parse(xliff_path)

    if not xlf.units:
        print("No translation units found.")
        return

    src = source_lang or xlf.sourcelanguage or "en"
    tgt = target_lang
    if not tgt:
        tgt = xlf.targetlanguage
    if not tgt:
        print(
            "No target language specified via --lang or XLIFF header", file=sys.stderr
        )
        return

    for unit in xlf.units:
        if not getattr(unit, "_source_locale_set", False):
            unit.xmlelement.set("source-language", src)
            unit.xmlelement.set("target-language", tgt)
            unit._source_locale_set = True

    untranslated = translator.get_untranslated(xlf)
    if not untranslated:
        print("All translation units are already translated.")
        translator.save(xlf, output_path)
        return

    _run_translation(
        store=xlf,
        units=untranslated,
        source_lang=src,
        target_lang=tgt,
        model_spec=model,
        translator=translator,
        output_path=output_path,
        context_file=context_file,
        context_length=context_length,
        verbose=verbose,
        progress_label=os.path.basename(output_path),
        api_key=api_key,
        api_host=api_host,
        temperature=temperature,
        progress=progress,
    )


def translate_xliff_dir(
    model: str,
    dir_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    context_file: str | None,
    context_length: int,
    jobs: int = 4,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
) -> None:
    """Translate all .xliff/.xlf files in a directory."""
    xliff_paths = [
        os.path.join(dir_path, entry)
        for entry in sorted(os.listdir(dir_path))
        if entry.endswith((".xliff", ".xlf"))
    ]
    if not xliff_paths:
        print("No .xliff/.xlf files found.")
        return

    max_workers = min(jobs, len(xliff_paths))
    progress = _build_progress()
    with progress, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                translate_xliff_file,
                model,
                xliff_path,
                source_lang,
                target_lang,
                verbose,
                xliff_path,
                context_file,
                context_length,
                api_key=api_key,
                api_host=api_host,
                temperature=temperature,
                progress=progress,
            ): xliff_path
            for xliff_path in xliff_paths
        }
        for future in as_completed(futures):
            future.result()
