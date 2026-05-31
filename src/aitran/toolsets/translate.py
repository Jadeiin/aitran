"""Translate toolset — wraps translate.py and review.py as orchestrator tools."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.toolsets import FunctionToolset

from aitran.review import review_file
from aitran.toolsets._base import (
    OrchestratorDeps,
    error_message,
    report_tool_outcome,
)
from aitran.translate import (
    translate_po,
    translate_po_dir,
    translate_xliff_dir,
    translate_xliff_file,
)

translate_toolset: FunctionToolset[OrchestratorDeps] = FunctionToolset()


def _noop_print(*_args, **_kwargs) -> None:
    return None


class _SilentProgress:
    """Minimal progress stub for orchestrator tool runs.

    The flow command already renders model output with Rich Live. Nested
    translation/review progress bars cause terminal redraw conflicts, so the
    orchestrator uses this no-op progress object instead.
    """

    console = SimpleNamespace(print=_noop_print)

    def add_task(self, *_args, **_kwargs) -> int:
        return 1

    def update(self, *_args, **_kwargs) -> None:
        return None


_SILENT_PROGRESS = _SilentProgress()


def _translate_kwargs(deps: OrchestratorDeps) -> dict:
    """Extract translate/review model config from deps.

    Returns:
        Dict with api_key, api_host, temperature.
    """
    return {
        "api_key": deps.translate_api_key,
        "api_host": deps.translate_api_host,
        "temperature": deps.translate_temperature,
    }


def _detect_format(path: str) -> str:
    """Detect file format from extension.

    Returns:
        'po', 'xliff', or 'dir'.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    p = Path(path)
    if p.is_dir():
        return "dir"
    ext = p.suffix.lower()
    if ext in (".po", ".pot"):
        return "po"
    if ext in (".xliff", ".xlf"):
        return "xliff"
    raise ValueError(f"Unsupported file format: {ext}")


@translate_toolset.tool(requires_approval=True)
async def translate_file(  # noqa: D417
    ctx: RunContext[OrchestratorDeps],
    path: str,
    source_lang: str = "en",
    target_lang: str = "",
    output_path: str | None = None,
    context_file: str | None = None,
    batch_size: int = 100,
    jobs: int = 4,
    order: str = "file",
    profile: str = "full",
) -> str:
    """Translate a PO or XLIFF file, or all files in a directory.

    The file format is auto-detected from the extension. For directories,
    all ``.po``, ``.xliff``, and ``.xlf`` files are translated.

    Args:
        path: Path to a ``.po``, ``.pot``, ``.xliff``, or ``.xlf`` file, or a
            directory containing ``.po``, ``.xliff``, or ``.xlf`` files.
        source_lang: Source language code (default: 'en').
        target_lang: Target language code. If empty, inferred from metadata.
        output_path: Output file path for single-file inputs. Keep the same
            translation suffix as the input. Defaults to in-place.
        context_file: Optional text file with extra translation context.
        batch_size: Max units per translation batch.
        jobs: Max files to translate concurrently for directory inputs.
        order: Unit ordering strategy: file, source, reference, or context.
        profile: Prompt detail level: fast or full.

    Returns:
        Confirmation message with file count.
    """
    try:
        fmt = _detect_format(path)
        model = ctx.deps.translate_model
        kwargs = _translate_kwargs(ctx.deps)
        resolved_output = output_path or path

        # translate_po/review_file use asyncio.run() internally,
        # so they must run in a thread to avoid nested event loop errors.
        if fmt == "po":
            await asyncio.to_thread(
                translate_po,
                model=model,
                po_path=path,
                source_lang=source_lang,
                target_lang=target_lang,
                verbose=False,
                output_path=resolved_output,
                context_file=context_file,
                batch_size=batch_size,
                progress=_SILENT_PROGRESS,
                order=order,
                profile=profile,
                **kwargs,
            )
            message = f"Translated PO file: {path}"
            report_tool_outcome(
                ctx.deps, tool_name="translate_file", message=message, ok=True
            )
            return message

        if fmt == "xliff":
            await asyncio.to_thread(
                translate_xliff_file,
                model=model,
                xliff_path=path,
                source_lang=source_lang,
                target_lang=target_lang,
                verbose=False,
                output_path=resolved_output,
                context_file=context_file,
                batch_size=batch_size,
                progress=_SILENT_PROGRESS,
                profile=profile,
                order=order,
                **kwargs,
            )
            message = f"Translated XLIFF file: {path}"
            report_tool_outcome(
                ctx.deps, tool_name="translate_file", message=message, ok=True
            )
            return message

        # Directory
        po_files = [f for f in os.listdir(path) if f.endswith(".po")]
        xliff_files = [f for f in os.listdir(path) if f.endswith((".xliff", ".xlf"))]
        if po_files:
            await asyncio.to_thread(
                translate_po_dir,
                model=model,
                dir_path=path,
                source_lang=source_lang,
                target_lang=target_lang,
                verbose=False,
                context_file=context_file,
                batch_size=batch_size,
                jobs=jobs,
                progress=_SILENT_PROGRESS,
                order=order,
                profile=profile,
                **kwargs,
            )
        if xliff_files:
            await asyncio.to_thread(
                translate_xliff_dir,
                model=model,
                dir_path=path,
                source_lang=source_lang,
                target_lang=target_lang,
                verbose=False,
                context_file=context_file,
                batch_size=batch_size,
                jobs=jobs,
                progress=_SILENT_PROGRESS,
                profile=profile,
                order=order,
                **kwargs,
            )
        total = len(po_files) + len(xliff_files)
        message = f"Translated {total} file(s) in {path}"
        report_tool_outcome(
            ctx.deps, tool_name="translate_file", message=message, ok=True
        )
        return message

    except Exception as e:  # noqa: BLE001
        message = error_message("Translate", e)
        report_tool_outcome(
            ctx.deps, tool_name="translate_file", message=message, ok=False
        )
        return message


@translate_toolset.tool(requires_approval=True)
async def review_translated_file(  # noqa: D417
    ctx: RunContext[OrchestratorDeps],
    path: str,
    source_lang: str = "en",
    target_lang: str = "",
    output_path: str | None = None,
    batch_size: int = 100,
    strict: bool = False,
    auto_fix: bool = False,
) -> str:
    """Review a translated PO or XLIFF file using QA checks and LLM.

    Runs rule-based QA checks, then sends problematic units to an LLM
    reviewer for verdict (pass/revise/reject).

    Args:
        path: Path to a translated ``.po``, ``.pot``, ``.xliff``, or ``.xlf``
            file.
        source_lang: Source language code (default: 'en').
        target_lang: Target language code. If empty, inferred from metadata.
        output_path: Output file path. Keep the same translation suffix as the
            input. Defaults to in-place review.
        batch_size: Max units per review batch.
        strict: If true, review every unit instead of only flagged ones.
        auto_fix: If true, write corrected targets back to the file.

    Returns:
        JSON string of review summary.
    """
    try:
        model = ctx.deps.translate_model
        kwargs = _translate_kwargs(ctx.deps)

        summary = await asyncio.to_thread(
            review_file,
            model=model,
            path=path,
            source_lang=source_lang,
            target_lang=target_lang,
            output_path=output_path or path,
            batch_size=batch_size,
            strict=strict,
            auto_fix=auto_fix,
            progress=_SILENT_PROGRESS,
            **kwargs,
        )
        message = json.dumps(summary, indent=2)
        report_tool_outcome(
            ctx.deps, tool_name="review_translated_file", message=message, ok=True
        )
        return message
    except Exception as e:  # noqa: BLE001
        message = error_message("Review", e)
        report_tool_outcome(
            ctx.deps, tool_name="review_translated_file", message=message, ok=False
        )
        return message
