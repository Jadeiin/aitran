"""Review pipeline: QA checks + LLM verdict on translated units."""

from __future__ import annotations

import asyncio
import sys
from contextlib import nullcontext
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

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from aitran.agents import ReviewBatch


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
) -> str:
    """Build XML input for the reviewer agent.

    Each unit includes source, target, and any QA errors.
    Units and qa_reports are parallel lists with matching indices.

    Returns:
        XML string for the reviewer agent prompt.
    """
    from pydantic_ai import format_as_xml

    from aitran.agents._base import safe_prompt_text

    items: list[dict] = []
    for u, report in zip(units, qa_reports, strict=True):
        d: dict = {
            "index": report.index,
            "source": safe_prompt_text(u.source),
            "target": safe_prompt_text(u.target),
        }
        if report.has_errors:
            d["qa-errors"] = "; ".join(
                f"[{e.severity}] {e.checker}: {e.message}" for e in report.errors
            )
        items.append(d)

    return format_as_xml(items, root_tag="review-batch", item_tag="unit")


def _filter_review_units(
    units: list,
    qa_reports: list[UnitQAReport],
    *,
    start_index: int,
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
        idx = start_index + i
        report = qa_by_index.get(idx)
        has_qa_errors = report is not None and report.has_errors
        is_fuzzy = getattr(u, "isfuzzy", lambda: False)()
        has_notes = bool(getattr(u, "getnotes", lambda: "")().strip())

        if strict or has_qa_errors or is_fuzzy or has_notes:
            filtered_reports.append(report or UnitQAReport(index=idx))
            filtered_units.append(u)

    return filtered_reports, filtered_units


async def _run_review_async(
    store,
    units: list,
    source_lang: str,
    target_lang: str,
    model_spec: str,
    translator,
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

    Processes units in serial batches (same batching strategy as
    translation), reviewing each batch after accumulation.

    Returns:
        Summary counts: ``{"pass": N, "revise": N, "reject": N}``.
    """
    base_url = (api_host.rstrip("/") + "/v1") if api_host else None
    agent = build_reviewer_agent(
        build_model(
            model_spec, api_key=api_key, base_url=base_url, temperature=temperature
        )
    )

    owns_progress = progress is None
    progress = progress or _build_progress()
    task_id = progress.add_task("Reviewing", total=len(units))

    summary: dict[str, int] = {"pass": 0, "revise": 0, "reject": 0}
    batch: list = []
    char_count = 0
    next_start_index = 1
    history: list = []
    qa_runner = QARunner(target_lang=target_lang)

    async def _review_batch(
        batch_units: list, start_idx: int
    ) -> tuple[list[ReviewedUnit], int]:
        """Review a single accumulated batch.

        Returns:
            Tuple of (review results, actual input XML char length).
        """
        qa_reports = qa_runner.check_units(batch_units, start_index=start_idx)

        review_reports, review_units = _filter_review_units(
            batch_units, qa_reports, start_index=start_idx, strict=strict
        )
        if not review_units:
            return [], 0

        input_xml = _build_review_input_xml(review_units, review_reports)
        deps = ReviewDeps(
            source_lang=source_lang,
            target_lang=target_lang,
            context="",
            expected_indices=tuple(r.index for r in review_reports),
        )
        result = await agent.run(
            input_xml, deps=deps, message_history=history
        )
        history.extend(result.new_messages())
        return result.output.units, len(input_xml)

    with progress if owns_progress else nullcontext():
        for unit in units:
            src_len = len(unit.source or "")
            if batch and char_count + src_len > context_length:
                reviewed, xml_len = await _review_batch(batch, next_start_index)
                for r in reviewed:
                    summary[r.verdict] = summary.get(r.verdict, 0) + 1
                summary["pass"] += len(batch) - len(reviewed)
                translator.apply_review_batch(
                    batch,
                    reviewed,
                    start_index=next_start_index,
                    auto_fix=auto_fix,
                )
                translator.save(store, output_path)
                progress.update(task_id, advance=len(batch))
                next_start_index += len(batch)
                batch = []
                char_count = xml_len

            batch.append(unit)
            char_count += src_len

    # Flush final batch
    if batch:
        reviewed, _ = await _review_batch(batch, next_start_index)
        for r in reviewed:
            summary[r.verdict] = summary.get(r.verdict, 0) + 1
        summary["pass"] += len(batch) - len(reviewed)
        translator.apply_review_batch(
            batch,
            reviewed,
            start_index=next_start_index,
            auto_fix=auto_fix,
        )
        translator.save(store, output_path)
        progress.update(task_id, advance=len(batch))

    if owns_progress:
        progress.stop()

    return summary


def build_default_reviewer(
    model_spec: str,
    *,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
) -> Agent[ReviewDeps, ReviewBatch]:
    """Build a reviewer agent from model spec strings.

    Returns:
        Configured reviewer agent.
    """
    base_url = (api_host.rstrip("/") + "/v1") if api_host else None
    model = build_model(
        model_spec, api_key=api_key, base_url=base_url, temperature=temperature
    )
    return build_reviewer_agent(model)


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
    from aitran.translate import PoTranslator

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

    units = [u for u in po_file.units if u.source and not u.isheader()]
    if not units:
        print("No entries to review.")
        return {"pass": 0, "revise": 0, "reject": 0}

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
