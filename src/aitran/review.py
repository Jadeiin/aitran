"""Review pipeline: QA checks + LLM verdict on translated units."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from aitran.agents import (
    ReviewDeps,
    ReviewedUnit,
    build_model,
    build_reviewer_agent,
)
from aitran.qa import QARunner, UnitQAReport
from aitran.translate import PoTranslator

if TYPE_CHECKING:
    from translate.storage import po


def _build_progress(console: Console | None = None) -> Progress:
    """Create a Rich progress renderer for review runs.

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


def _build_review_input_xml(
    units: list,
    qa_reports: list[UnitQAReport],
    start_index: int,
) -> str:
    """Build XML input for the reviewer agent.

    Each unit includes source, target, and any QA errors.

    Returns:
        XML string for the reviewer agent prompt.
    """
    from pydantic_ai import format_as_xml

    from aitran.agents._base import safe_prompt_text

    qa_by_index = {r.index: r for r in qa_reports}
    items: list[dict] = []
    for i, u in enumerate(units):
        idx = start_index + i
        d: dict = {
            "index": idx,
            "source": safe_prompt_text(u.source),
            "target": safe_prompt_text(u.target),
        }
        report = qa_by_index.get(idx)
        if report and report.has_errors:
            d["qa-errors"] = "; ".join(
                f"[{e.severity}] {e.checker}: {e.message}" for e in report.errors
            )
        items.append(d)

    return format_as_xml(items, root_tag="review-batch", item_tag="unit")


def _filter_review_units(
    units: list,
    qa_reports: list[UnitQAReport],
    *,
    strict: bool,
) -> tuple[list[UnitQAReport], list]:
    """Filter units for review.

    Returns:
        Tuple of (qa_reports_to_review, units_to_review) — only units
        that need LLM review are included.
    """
    qa_by_index = {r.index: r for r in qa_reports}
    filtered_reports: list[UnitQAReport] = []
    filtered_units: list = []
    for i, u in enumerate(units):
        idx = i + 1  # 1-based
        report = qa_by_index.get(idx)
        has_qa_errors = report is not None and report.has_errors
        is_fuzzy = getattr(u, "isfuzzy", lambda: False)()
        has_notes = bool(getattr(u, "getnotes", lambda: "")().strip())

        if strict or has_qa_errors or is_fuzzy or has_notes:
            filtered_reports.append(report or UnitQAReport(index=idx))
            filtered_units.append(u)

    return filtered_reports, filtered_units


async def _run_review_async(
    store: po.pofile,
    units: list,
    source_lang: str,
    target_lang: str,
    model_spec: str,
    translator: PoTranslator,
    output_path: str,
    context_length: int,
    *,
    auto_fix: bool = False,
    strict: bool = False,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> dict[str, int]:
    """Run QA + LLM review on translated units.

    Returns:
        Summary counts: ``{"pass": N, "revise": N, "reject": N}``.
    """
    import logfire

    # 1. Run QA on all units
    with logfire.span(
        "qa-check",
        unit_count=len(units),
        target_lang=target_lang,
    ):
        qa_runner = QARunner(target_lang=target_lang)
        qa_reports = qa_runner.check_units(units, start_index=1)

    # 2. Filter units needing LLM review
    review_reports, review_units = _filter_review_units(
        units, qa_reports, strict=strict
    )

    if not review_units:
        translator.save(store, output_path)
        return {"pass": len(units), "revise": 0, "reject": 0}

    # 3. Build reviewer agent
    base_url = (api_host.rstrip("/") + "/v1") if api_host else None
    agent = build_reviewer_agent(
        build_model(
            model_spec, api_key=api_key, base_url=base_url, temperature=temperature
        )
    )

    owns_progress = progress is None
    progress = progress or _build_progress()
    task_id = progress.add_task("Reviewing", total=len(review_units))

    # 4. Map review-unit positions back to original unit list
    review_to_original: list[int] = []
    for i, u in enumerate(units):
        idx = i + 1
        report = {r.index: r for r in qa_reports}.get(idx)
        has_qa_errors = report is not None and report.has_errors
        is_fuzzy = getattr(u, "isfuzzy", lambda: False)()
        has_notes = bool(getattr(u, "getnotes", lambda: "")().strip())
        if strict or has_qa_errors or is_fuzzy or has_notes:
            review_to_original.append(i)

    # 5. Batch and review
    batch: list = []
    batch_reports: list[UnitQAReport] = []
    char_count = 0
    next_start_index = 1
    summary: dict[str, int] = {"pass": 0, "revise": 0, "reject": 0}

    async def _review_batch(
        batch_units: list,
        batch_qa: list[UnitQAReport],
        start_idx: int,
    ) -> list[ReviewedUnit]:
        input_xml = _build_review_input_xml(batch_units, batch_qa, start_idx)
        deps = ReviewDeps(
            source_lang=source_lang,
            target_lang=target_lang,
            context="",
            expected_indices=tuple(range(start_idx, start_idx + len(batch_units))),
        )
        result = await agent.run(input_xml, deps=deps)
        return result.output.units

    for unit, report in zip(review_units, review_reports, strict=True):
        src_len = len(unit.source or "")
        if batch and char_count + src_len > context_length:
            with logfire.span(
                "review-batch",
                batch_size=len(batch),
                start_index=next_start_index,
            ):
                results = await _review_batch(
                    batch, batch_reports, next_start_index
                )
            for reviewed in results:
                summary[reviewed.verdict] = summary.get(reviewed.verdict, 0) + 1
            translator.apply_review_batch(batch, results, auto_fix=auto_fix)
            progress.update(task_id, advance=len(batch))
            next_start_index += len(batch)
            batch = []
            batch_reports = []
            char_count = 0

        batch.append(unit)
        batch_reports.append(report)
        char_count += src_len

    # Flush final batch
    if batch:
        with logfire.span(
            "review-batch",
            batch_size=len(batch),
            start_index=next_start_index,
        ):
            results = await _review_batch(batch, batch_reports, next_start_index)
        for reviewed in results:
            summary[reviewed.verdict] = summary.get(reviewed.verdict, 0) + 1
        translator.apply_review_batch(batch, results, auto_fix=auto_fix)
        progress.update(task_id, advance=len(batch))

    # Count passes (units not sent to LLM)
    summary["pass"] += len(units) - len(review_units)

    translator.save(store, output_path)
    if owns_progress:
        progress.stop()

    return summary


def review_po(
    model: str,
    po_path: str,
    source_lang: str,
    target_lang: str,
    output_path: str,
    context_length: int,
    *,
    strict: bool = False,
    auto_fix: bool = False,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
) -> dict[str, int]:
    """Review a single PO file.

    Returns:
        Summary counts: ``{"pass": N, "revise": N, "reject": N}``.
    """
    import logfire

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
        return {"pass": 0, "revise": 0, "reject": 0}

    # Get all translated units (not just untranslated)
    units = [u for u in po_file.units if u.source and not u.isheader()]
    if not units:
        print("No entries to review.")
        return {"pass": 0, "revise": 0, "reject": 0}

    with logfire.span(
        "review-po",
        po_path=po_path,
        source_lang=source_lang,
        target_lang=target_lang,
        unit_count=len(units),
        model=model,
        strict=strict,
        auto_fix=auto_fix,
    ):
        return asyncio.run(
            _run_review_async(
                store=po_file,
                units=units,
                source_lang=source_lang,
                target_lang=target_lang,
                model_spec=model,
                translator=translator,
                output_path=output_path,
                context_length=context_length,
                auto_fix=auto_fix,
                strict=strict,
                api_key=api_key,
                api_host=api_host,
                temperature=temperature,
            )
        )
