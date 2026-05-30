"""Review pipeline: QA checks + LLM verdict on translated units."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import UnexpectedModelBehavior

from aitran.agents import (
    ReviewDeps,
    ReviewedUnit,
    build_model,
    build_review_input_xml,
    build_reviewer_agent,
)
from aitran.agents._base import fmt_base_url
from aitran.qa import QARunner, UnitQAReport
from aitran.translate import _build_progress, _emit_status

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from rich.progress import Progress

    from aitran.agents import ReviewBatch


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

        if strict:
            filtered_reports.append(report or UnitQAReport(index=idx))
            filtered_units.append(u)
        else:
            is_fuzzy = getattr(u, "isfuzzy", lambda: False)()
            has_notes = bool(getattr(u, "getnotes", lambda: "")().strip())
            if has_qa_errors or is_fuzzy or has_notes:
                filtered_reports.append(report or UnitQAReport(index=idx))
                filtered_units.append(u)

    return filtered_reports, filtered_units


def _pre_filter_by_markers(
    units: list,
    *,
    start_index: int,
) -> tuple[list[UnitQAReport], list, list[int]]:
    """Split units into fuzzy/note-marked candidates and the rest.

    Units with fuzzy flags or translator notes are always review-worthy
    without running expensive QA checks.  The remaining indices need QA.

    Returns:
        Tuple of (reports, units, indices_needing_qa) — the first two lists
        contain units that bypass QA; the third is 1-based indices of units
        that still need QA checks.
    """
    marked_reports: list[UnitQAReport] = []
    marked_units: list = []
    qa_indices: list[int] = []
    for i, u in enumerate(units):
        idx = start_index + i
        is_fuzzy = getattr(u, "isfuzzy", lambda: False)()
        has_notes = bool(getattr(u, "getnotes", lambda: "")().strip())
        if is_fuzzy or has_notes:
            marked_reports.append(UnitQAReport(index=idx))
            marked_units.append(u)
        else:
            qa_indices.append(idx)
    return marked_reports, marked_units, qa_indices


async def _review_batch(
    agent,
    units: list,
    qa_reports: list[UnitQAReport],
    deps: ReviewDeps,
    history: list,
    on_progress=None,
) -> list[ReviewedUnit]:
    """Stream one review batch through the agent.

    Returns:
        List of ReviewedUnit for units that need revise or reject handling.
    """
    user_msg = build_review_input_xml(units, qa_reports)
    valid_indices = set(deps.expected_indices)
    seen: set[int] = set()

    async with agent.run_stream(
        user_msg,
        deps=deps,
        message_history=history,
    ) as run:
        async for partial in run.stream_output(debounce_by=0.1):
            for result in partial.units:
                if result.index in seen or result.index not in valid_indices:
                    continue
                seen.add(result.index)
                if on_progress:
                    on_progress(result)
        final = await run.get_output()
        history.extend(run.new_messages())

    return final.units


async def _run_review_async(
    store,
    units: list,
    source_lang: str,
    target_lang: str,
    model_spec: str,
    translator,
    output_path: str,
    batch_size: int,
    *,
    auto_fix: bool = False,
    strict: bool = False,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> dict[str, int]:
    """Run QA + LLM review on translated units.

    Runs QA globally, filters to review-worthy units, then sends them
    to the agent in fixed-size batches within the same conversation.

    Returns:
        Summary counts: ``{"pass": N, "revise": N, "reject": N}``.
    """
    base_url = fmt_base_url(api_host)
    agent = build_reviewer_agent(
        build_model(
            model_spec, api_key=api_key, base_url=base_url, temperature=temperature
        )
    )

    owns_progress = progress is None
    progress = progress or _build_progress()

    # In non-strict mode, pre-filter by fuzzy/notes markers (cheap) before
    # running expensive QA checks — only units without markers need QA.
    qa_runner = QARunner(target_lang=target_lang)
    if strict:
        qa_reports = qa_runner.check_units(units, start_index=1)
        review_reports, review_units = _filter_review_units(
            units, qa_reports, start_index=1, strict=True
        )
    else:
        marked_reports, marked_units, qa_indices = _pre_filter_by_markers(
            units, start_index=1
        )
        if qa_indices:
            qa_units = [units[idx - 1] for idx in qa_indices]
            qa_reports = qa_runner.check_units(qa_units, start_index=qa_indices[0])
            qa_only_reports = [r for r in qa_reports if r.has_errors]
            qa_only_units = [units[r.index - 1] for r in qa_only_reports]
        else:
            qa_only_reports = []
            qa_only_units = []
        review_reports = marked_reports + qa_only_reports
        review_units = marked_units + qa_only_units
    review_count = len(review_units)
    total_count = len(units)

    summary: dict[str, int] = {
        "pass": total_count - review_count,
        "revise": 0,
        "reject": 0,
        "skip": 0,
    }
    if not review_units:
        _emit_status(
            f"Reviewed {total_count} units, all clean.",
            progress=progress,
        )
        translator.save(store, output_path)
        return summary

    task_id = progress.add_task("Reviewing", total=review_count)
    history: list = []
    unit_by_index = dict(enumerate(units, start=1))
    global_done = 0
    batch_streamed: set[int] = set()
    saved_indices: set[int] = set()

    def on_review_done(result: ReviewedUnit) -> None:
        nonlocal global_done
        idx = result.index
        if idx in saved_indices or idx in batch_streamed:
            return
        batch_streamed.add(idx)
        global_done += 1
        progress.update(task_id, completed=global_done)

    def _commit_batch(batch_reports: list[UnitQAReport], reviewed: list[ReviewedUnit]):
        nonlocal global_done
        reviewed_indices = {r.index for r in reviewed}
        for result in reviewed:
            on_review_done(result)
        pass_count = len(batch_reports) - len(reviewed_indices)
        global_done += pass_count
        saved_indices.update(r.index for r in batch_reports)
        batch_streamed.clear()
        if pass_count:
            progress.update(task_id, completed=global_done)

    def _rollback_batch() -> None:
        nonlocal global_done
        global_done -= len(batch_streamed)
        batch_streamed.clear()
        progress.update(task_id, completed=global_done)

    next_start = 0
    batch_retries = 0
    BATCH_MAX_RETRIES = 3
    console = progress.console

    async def _flush_batch() -> None:
        nonlocal batch_retries, next_start
        batch_units = review_units[next_start : next_start + batch_size]
        batch_reports = review_reports[next_start : next_start + batch_size]
        deps = ReviewDeps(
            source_lang=source_lang,
            target_lang=target_lang,
            context="",
            expected_indices=tuple(r.index for r in batch_reports),
        )
        while True:
            try:
                reviewed = await _review_batch(
                    agent,
                    batch_units,
                    batch_reports,
                    deps,
                    history,
                    on_review_done,
                )
                for r in reviewed:
                    summary[r.verdict] = summary.get(r.verdict, 0) + 1
                summary["pass"] += len(batch_units) - len(reviewed)
                translator.apply_review_batch(
                    store,
                    unit_by_index,
                    reviewed,
                    auto_fix=auto_fix,
                )
                translator.save(store, output_path)
                _commit_batch(batch_reports, reviewed)
                next_start += len(batch_units)
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
                    f"Skipping {len(batch_units)} unit(s).[/]"
                )
                summary["skip"] += len(batch_units)
                saved_indices.update(r.index for r in batch_reports)
                batch_streamed.clear()
                translator.save(store, output_path)
                next_start += len(batch_units)
                batch_retries = 0
                return

    with progress if owns_progress else nullcontext():
        while next_start < review_count:
            await _flush_batch()

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
    base_url = fmt_base_url(api_host)
    model = build_model(
        model_spec, api_key=api_key, base_url=base_url, temperature=temperature
    )
    return build_reviewer_agent(model)


def review_file(
    model: str,
    path: str,
    source_lang: str,
    target_lang: str,
    output_path: str,
    batch_size: int,
    *,
    strict: bool = False,
    auto_fix: bool = False,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    progress: Progress | None = None,
) -> dict[str, int]:
    """Review a single PO or XLIFF file.

    Returns:
        Summary counts: ``{"pass": N, "revise": N, "reject": N}``.
    """
    from aitran.translate import PoTranslator, XliffTranslator

    empty = {"pass": 0, "revise": 0, "reject": 0, "skip": 0}
    is_po = path.endswith((".po", ".pot"))

    if is_po:
        translator = PoTranslator()
        store = translator.parse(path)

        if not target_lang:
            inferred_lang = translator.get_target_language(store)
            if inferred_lang:
                target_lang = inferred_lang
        if not target_lang:
            _emit_status(
                "No target language specified via --lang or PO header",
                progress=progress,
                stderr=True,
            )
            return empty

        src = source_lang
        units = [u for u in store.units if u.source and not u.isheader() and u.target]
    else:
        translator = XliffTranslator()
        store = translator.parse(path)

        if not target_lang:
            target_lang = store.targetlanguage or ""
        if not target_lang:
            _emit_status(
                "No target language specified via --lang or XLIFF header",
                progress=progress,
                stderr=True,
            )
            return empty

        src = source_lang or store.sourcelanguage or "en"
        units = [
            u
            for u in store.units
            if (u.source or "").strip() and (u.target or "").strip()
        ]

    if not units:
        _emit_status("No translated entries to review.", progress=progress)
        return empty

    return asyncio.run(
        _run_review_async(
            store=store,
            units=units,
            source_lang=src,
            target_lang=target_lang,
            model_spec=model,
            translator=translator,
            output_path=output_path,
            batch_size=batch_size,
            auto_fix=auto_fix,
            strict=strict,
            api_key=api_key,
            api_host=api_host,
            temperature=temperature,
            progress=progress,
        )
    )
