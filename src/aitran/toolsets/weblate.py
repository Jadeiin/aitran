"""Weblate toolset — wraps weblate.py API functions as orchestrator tools."""

from __future__ import annotations

import json

from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.toolsets import FunctionToolset

from aitran.toolsets._base import OrchestratorDeps  # noqa: TC001
from aitran.weblate import (
    download_translation as weblate_download,
)
from aitran.weblate import (
    get_stats as weblate_stats,
)
from aitran.weblate import (
    list_objects as weblate_list_objects,
)
from aitran.weblate import (
    upload_translation as weblate_upload,
)

weblate_toolset: FunctionToolset[OrchestratorDeps] = FunctionToolset()


def _require_url(deps: OrchestratorDeps) -> str:
    if not deps.weblate_url:
        raise ValueError("Weblate URL not configured. Set AITRAN_WEBLATE_URL.")
    return deps.weblate_url


def _require_token(deps: OrchestratorDeps) -> str:
    if not deps.weblate_token:
        raise ValueError("Weblate token not configured. Set AITRAN_WEBLATE_TOKEN.")
    return deps.weblate_token


@weblate_toolset.tool
async def list_objects(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    object_path: str | None = None,
) -> str:
    """List Weblate projects or child objects at a path.

    Args:
        object_path: Optional Weblate object path
            (e.g. 'project/component/lang'). Omit to list all projects.

    Returns:
        JSON string of object summaries.
    """
    items = weblate_list_objects(
        url=_require_url(ctx.deps),
        token=_require_token(ctx.deps),
        object_path=object_path,
    )
    if not items:
        return "No objects found."
    summary = []
    for item in items:
        if hasattr(item, "get_data"):
            data = item.get_data()
            entry = {k: v for k, v in data.items() if not k.startswith("_")}
        elif isinstance(item, dict):
            entry = {k: v for k, v in item.items() if not k.startswith("_")}
        else:
            entry = {"value": str(item)}
        summary.append(entry)
    return json.dumps(summary[:50], ensure_ascii=False, indent=2, default=str)


@weblate_toolset.tool
async def get_stats(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    object_path: str,
) -> str:
    """Get statistics for a Weblate project, component, or translation.

    Args:
        object_path: Weblate object path (e.g. 'project' or 'project/component').

    Returns:
        JSON string of statistics.
    """
    stats = weblate_stats(
        url=_require_url(ctx.deps),
        token=_require_token(ctx.deps),
        object_path=object_path,
    )
    # wlc SDK objects are dict subclasses — convert to plain dicts
    # so json.dumps can serialize the API data.
    if isinstance(stats, list):
        cleaned = [
            {k: v for k, v in s.get_data().items() if not k.startswith("_")}
            if hasattr(s, "get_data")
            else {k: v for k, v in s.items() if not k.startswith("_")}
            if isinstance(s, dict)
            else str(s)
            for s in stats
        ]
        return json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
    if hasattr(stats, "get_data"):
        data = stats.get_data()
        cleaned = {k: v for k, v in data.items() if not k.startswith("_")}
        return json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
    if isinstance(stats, dict):
        cleaned = {k: v for k, v in stats.items() if not k.startswith("_")}
        return json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)
    return str(stats)


@weblate_toolset.tool(requires_approval=True)
async def download_translation(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    object_path: str,
    output_path: str,
    untranslated_only: bool = False,
) -> str:
    """Download a translation file from Weblate.

    Args:
        object_path: Weblate translation path
            (<project>/<component>/<language>).
        output_path: Local file path to save the download. Must end with
            ``.po``, ``.xliff``, or ``.xlf``.
        untranslated_only: If true, download only untranslated strings.

    Returns:
        Confirmation message with output path.
    """
    weblate_download(
        url=_require_url(ctx.deps),
        token=_require_token(ctx.deps),
        object_path=object_path,
        output_path=output_path,
        download_format=None,
        untranslated_only=untranslated_only,
    )
    return f"Downloaded to {output_path}"


@weblate_toolset.tool(requires_approval=True)
async def upload_translation(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    object_path: str,
    file_path: str,
    method: str = "translate",
    conflicts: str = "ignore",
) -> str:
    """Upload a translation file to Weblate.

    Args:
        object_path: Weblate translation path
            (<project>/<component>/<language>).
        file_path: Local translation file to upload. Must end with ``.po``,
            ``.xliff``, or ``.xlf``.
        method: Upload method
            (translate, approve, suggest, fuzzy, replace, source, add).
        conflicts: Existing-translation handling: ignore, replace-translated,
            or replace-approved. Use replace-translated when uploading corrections.

    Returns:
        Confirmation message.
    """
    result = weblate_upload(
        url=_require_url(ctx.deps),
        token=_require_token(ctx.deps),
        object_path=object_path,
        file_path=file_path,
        method=method,
        fuzzy=None,
        conflicts=conflicts,
    )
    return json.dumps(
        {"status": "uploaded", "file_path": file_path, "result": result},
        ensure_ascii=False,
        indent=2,
        default=str,
    )
