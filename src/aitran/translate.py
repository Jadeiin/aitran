"""Core translation engine built on a Pydantic AI agent."""

from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, ClassVar

from pydantic_ai.exceptions import UnexpectedModelBehavior
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from translate.misc import quote, xml_helpers
from translate.misc.multistring import multistring
from translate.storage import po, xliff

from aitran.agents import (
    ReviewedUnit,
    TranslationDeps,
    build_model,
    build_translation_input_xml,
    build_translator_agent,
)
from aitran.dicts import find_matching_entries

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aitran.agents import TranslatedUnit

_LEGACY_LANGUAGE_CODES = {
    "zh_Hans": "zh_CN",
    "zh_Hant": "zh_TW",
    "zh_Hans_SG": "zh_SG",
    "zh_Hant_HK": "zh_HK",
}


_XML_ENTITY_CODEPOINTS = {
    "amp": ord("&"),
    "lt": ord("<"),
    "gt": ord(">"),
}


def _decode_serialized_markup(source: str, target: str) -> str:
    """Reverse prompt XML escaping only for entities introduced by source text.

    Returns:
        Target text with prompt serialization entities decoded only when needed.
    """
    entity_codepoints = {
        name: codepoint
        for name, codepoint in _XML_ENTITY_CODEPOINTS.items()
        if chr(codepoint) in source
    }
    if not entity_codepoints:
        return target
    return quote.entitydecode(target, entity_codepoints)


def _read_context(context_file: str | None) -> str:
    if not context_file:
        return ""
    with open(context_file, encoding="utf-8") as f:
        return f.read().strip()


def _last_translator() -> str:
    try:
        package_version = version("aitran")
    except PackageNotFoundError:
        package_version = "unknown"
    return f"aitran v{package_version}"


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
    def get_target_language(po_file: po.pofile) -> str | None:
        """Infer target language from PO metadata.

        Returns:
            Target language string, or None if no language can be inferred.
        """
        if target_language := po_file.gettargetlanguage():
            return target_language

        language_header = po_file.parseheader().get("Language", "").strip()
        return _LEGACY_LANGUAGE_CODES.get(language_header)

    @staticmethod
    def get_untranslated(po_file: po.pofile) -> list[po.pounit]:
        """Return units that need translation (empty target or fuzzy).

        Returns:
            List of untranslated or fuzzy PO units.
        """
        result: list[po.pounit] = []
        plural_tags = po_file.get_plural_tags()
        for unit in po_file.units:
            if unit.isheader():
                continue
            if unit.istranslated() and not unit.isfuzzy():
                # istranslated only checks singular form;
                # verify all plural forms are present and non-empty.
                if unit.hasplural() and len(plural_tags) > 1:
                    targets = (
                        unit.target.strings
                        if hasattr(unit.target, "strings")
                        else [unit.target]
                    )
                    if len(targets) < len(plural_tags) or any(
                        not t.strip() for t in targets
                    ):
                        result.append(unit)
                        continue
                continue
            result.append(unit)
        return result

    @staticmethod
    def apply_batch(
        po_file: po.pofile,
        units: list[po.pounit],
        results: list[TranslatedUnit],
    ) -> None:
        """Apply a batch of agent results."""
        plural_tags = po_file.get_plural_tags()
        for unit, result in zip(units, results, strict=True):
            cleaned = [xml_helpers.valid_chars_only(t) for t in result.targets]
            if unit.hasplural():
                if len(cleaned) != len(plural_tags):
                    result.fuzzy = True
                target = po.pounit.sync_plural_count(
                    multistring(cleaned),
                    plural_tags,
                )
            else:
                target = cleaned[0]
            unit.target = target
            unit.markfuzzy(result.fuzzy)
            if result.note:
                unit.addnote(result.note, origin="translator")

    @staticmethod
    def apply_review_batch(
        po_file: po.pofile,
        units_by_index: Mapping[int, po.pounit],
        results: list[ReviewedUnit],
        *,
        auto_fix: bool = False,
    ) -> None:
        """Apply review results to PO units.

        Without *auto_fix*, only marks ``revise``/``reject`` entries as fuzzy
        with a review note.  With *auto_fix*, also writes the corrected target
        and clears the fuzzy marker.
        """
        for result in results:
            unit = units_by_index[result.index]
            if auto_fix and result.corrected is not None:
                corrected = _decode_serialized_markup(
                    str(unit.source),
                    xml_helpers.valid_chars_only(result.corrected),
                )
                if unit.hasplural():
                    existing = (
                        unit.target.strings
                        if hasattr(unit.target, "strings")
                        else [str(unit.target)]
                    )
                    forms = [corrected, *existing[1:]]
                    unit.target = po.pounit.sync_plural_count(
                        multistring(forms),
                        po_file.get_plural_tags(),
                    )
                else:
                    unit.target = corrected
                unit.markfuzzy(unit.hasplural())
            else:
                unit.markfuzzy(True)
            if result.note:
                unit.addnote(
                    f"(review) {result.verdict}: {result.note}",
                    origin="translator",
                )

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
    _DONE_STATES: ClassVar[set[str]] = {"final", "signed-off", "translated"}

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

            if state.startswith("needs-") or state == "new":
                result.append(unit)
                continue
            if not target:
                result.append(unit)
                continue
            if state in cls._DONE_STATES:
                continue
            if target == source:
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
            unit.settarget(xml_helpers.valid_chars_only(result.targets[0]))
            if result.fuzzy:
                unit.markreviewneeded()
            else:
                unit.marktranslated()
            if result.note:
                unit.addnote(result.note, origin="translator")

    @staticmethod
    def apply_review_batch(
        _store,
        units_by_index: Mapping[int, xliff.xliffunit],
        results: list[ReviewedUnit],
        *,
        auto_fix: bool = False,
    ) -> None:
        """Apply review results to XLIFF units.

        Without *auto_fix*, only marks ``revise``/``reject`` entries as
        needs-review with a note.  With *auto_fix*, also writes the corrected
        target and marks translated.
        """
        for result in results:
            unit = units_by_index[result.index]
            if auto_fix and result.corrected is not None:
                unit.settarget(
                    _decode_serialized_markup(
                        str(unit.source),
                        xml_helpers.valid_chars_only(result.corrected),
                    )
                )
                unit.marktranslated()
            else:
                unit.markreviewneeded()
            if result.note:
                unit.addnote(
                    f"(review) {result.verdict}: {result.note}",
                    origin="translator",
                )

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
    *,
    profile: str = "full",
) -> list[TranslatedUnit]:
    """Stream one batch through the agent.

    Returns:
        List of TranslatedUnit aligned with the input units list.
    """
    user_msg = build_translation_input_xml(
        units,
        start_index,
        profile=profile,
        plural_tags=deps.plural_tags,
    )
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
                if not t.targets:
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
        # Reverse XML escaping applied by format_as_xml only when the source
        # had raw markup. Already-escaped source strings must remain escaped.
        raw = units[i].source
        source_strings = raw.strings if hasattr(raw, "strings") else [str(raw)]
        if (
            len(tu.targets) == 1
            and len(source_strings) > 1
            and deps.plural_tags
            and len(deps.plural_tags) == 1
        ):
            # One-form plural (e.g. Chinese): decode against all source forms.
            combined = " ".join(str(s) for s in source_strings)
            tu.targets = [_decode_serialized_markup(combined, tu.targets[0])]
        else:
            tu.targets = [
                _decode_serialized_markup(
                    str(source_strings[min(j, len(source_strings) - 1)]),
                    t,
                )
                for j, t in enumerate(tu.targets)
            ]
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
    batch_size: int,
    verbose: bool,
    progress_label: str,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
    profile: str = "full",
    plural_tags: list[str] | None = None,
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
            display_target = result.targets[0] if result.targets else ""
            tgt_short = display_target[:60] + ("…" if len(display_target) > 60 else "")
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
    next_start_index = 1
    batch_retries = 0
    BATCH_MAX_RETRIES = 3

    async def _flush_batch() -> None:
        """Flush current batch and apply translations."""
        nonlocal batch_retries
        nonlocal next_start_index, batch
        while True:
            deps = TranslationDeps(
                source_lang=source_lang,
                target_lang=target_lang,
                context=context,
                dict_entries=dict_entries,
                expected_indices=tuple(
                    range(next_start_index, next_start_index + len(batch))
                ),
                plural_tags=plural_tags,
            )
            try:
                results = await _translate_batch(
                    agent,
                    batch,
                    next_start_index,
                    deps,
                    history,
                    on_unit_done,
                    profile=profile,
                )
                translator.apply_batch(store, batch, results)
                translator.save(store, output_path)
                _commit_batch()
                next_start_index += len(batch)
                batch = []
                batch_retries = 0
                return
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
                batch_retries = 0
                return

    with progress if owns_progress else nullcontext():
        for unit in units:
            if len(batch) >= batch_size:
                await _flush_batch()
            batch.append(unit)

    # Flush remaining units
    if batch:
        await _flush_batch()


def _run_translation(
    store,
    units: list,
    source_lang: str,
    target_lang: str,
    model_spec: str,
    translator,
    output_path: str,
    context_file: str | None,
    batch_size: int,
    verbose: bool,
    progress_label: str,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
    profile: str = "full",
    plural_tags: list[str] | None = None,
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
            batch_size=batch_size,
            verbose=verbose,
            progress_label=progress_label,
            api_key=api_key,
            api_host=api_host,
            temperature=temperature,
            progress=progress,
            profile=profile,
            plural_tags=plural_tags,
        )
    )


def _order_units(units: list, order: str) -> list:
    """Reorder translation units by the chosen strategy.

    ``"file"`` (default) preserves the original PO file order.  ``"source"``
    sorts alphabetically for dedup / cache-hit friendliness.  ``"reference"``
    groups by the first source-code location so strings from the same file
    are translated together.  ``"context"`` groups by ``msgctxt`` so strings
    sharing a disambiguation context stay adjacent.

    Returns:
        Reordered list of units.
    """
    if order == "file" or not units:
        return units

    indexed = list(enumerate(units))
    if order == "source":
        indexed.sort(key=lambda item: item[1].source)
    elif order == "reference":

        def _reference_key(item):
            u = item[1]
            locs = u.getlocations() if hasattr(u, "getlocations") else []
            return (locs[0] if locs else "", item[0])

        indexed.sort(key=_reference_key)
    elif order == "context":

        def _context_key(item):
            u = item[1]
            getctx = getattr(u, "getcontext", None)
            ctx = getctx() if callable(getctx) else ""
            return (ctx, item[0])

        indexed.sort(key=_context_key)

    return [u for _, u in indexed]


def translate_po(
    model: str,
    po_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    output_path: str,
    context_file: str | None,
    batch_size: int,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
    order: str = "file",
    profile: str = "full",
) -> None:
    """Translate a single PO file."""
    translator = PoTranslator()
    po_file = translator.parse(po_path)

    if not target_lang:
        inferred_lang = translator.get_target_language(po_file)
        if inferred_lang:
            target_lang = inferred_lang
    if not target_lang:
        print(
            "No target language specified via --lang or PO header",
            file=sys.stderr,
        )
        return

    untranslated = translator.get_untranslated(po_file)
    if not untranslated:
        print("All entries already translated.")
        translator.save(po_file, output_path)
        return

    po_file.updateheader(
        add=True,
        **{
            "Last-Translator": _last_translator(),
            "Language": target_lang,
        },
    )

    untranslated = _order_units(untranslated, order)

    plural_tags = (
        po_file.get_plural_tags() if hasattr(po_file, "get_plural_tags") else None
    )

    _run_translation(
        store=po_file,
        units=untranslated,
        source_lang=source_lang,
        target_lang=target_lang,
        model_spec=model,
        translator=translator,
        output_path=output_path,
        context_file=context_file,
        batch_size=batch_size,
        verbose=verbose,
        progress_label=os.path.basename(output_path),
        api_key=api_key,
        api_host=api_host,
        temperature=temperature,
        progress=progress,
        profile=profile,
        plural_tags=plural_tags,
    )


def translate_po_dir(
    model: str,
    dir_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    context_file: str | None,
    batch_size: int,
    jobs: int = 4,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    order: str = "file",
    profile: str = "full",
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
                batch_size,
                api_key=api_key,
                api_host=api_host,
                temperature=temperature,
                progress=progress,
                order=order,
                profile=profile,
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
    batch_size: int,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
    profile: str = "full",
    order: str = "file",
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

    untranslated = _order_units(untranslated, order)

    _run_translation(
        store=xlf,
        units=untranslated,
        source_lang=src,
        target_lang=tgt,
        model_spec=model,
        translator=translator,
        output_path=output_path,
        context_file=context_file,
        batch_size=batch_size,
        verbose=verbose,
        progress_label=os.path.basename(output_path),
        api_key=api_key,
        api_host=api_host,
        temperature=temperature,
        progress=progress,
        profile=profile,
    )


def translate_xliff_dir(
    model: str,
    dir_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    context_file: str | None,
    batch_size: int,
    jobs: int = 4,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    profile: str = "full",
    order: str = "file",
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
                batch_size,
                api_key=api_key,
                api_host=api_host,
                temperature=temperature,
                progress=progress,
                profile=profile,
                order=order,
            ): xliff_path
            for xliff_path in xliff_paths
        }
        for future in as_completed(futures):
            future.result()
