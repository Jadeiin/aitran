"""Crowdin toolset — wraps crowdin.py API functions as orchestrator tools."""

from __future__ import annotations

from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.toolsets import FunctionToolset

from aitran.crowdin import (
    download_translation as crowdin_download,
)
from aitran.crowdin import (
    get_progress as crowdin_progress,
)
from aitran.crowdin import (
    list_files as crowdin_list_files,
)
from aitran.crowdin import (
    list_languages as crowdin_list_languages,
)
from aitran.crowdin import (
    list_projects as crowdin_list_projects,
)
from aitran.crowdin import (
    upload_translation as crowdin_upload,
)
from aitran.toolsets._base import (
    OrchestratorDeps,
    error_message,
    report_tool_outcome,
    summarize_list,
    summarize_progress,
)

crowdin_toolset: FunctionToolset[OrchestratorDeps] = FunctionToolset()


def _require_token(deps: OrchestratorDeps) -> str:
    if not deps.crowdin_token:
        raise ValueError("Crowdin token not configured. Set AITRAN_CROWDIN_TOKEN.")
    return deps.crowdin_token


def _project_kwargs(project: str) -> dict:
    """Build project ID/name kwargs for crowdin API calls.

    Returns:
        Dict with project_id or project key.
    """
    if project.isdigit():
        return {"project_id": int(project), "project": None}
    return {"project_id": None, "project": project}


def _report(
    ctx: RunContext[OrchestratorDeps], tool_name: str, message: str, ok: bool
) -> str:
    report_tool_outcome(ctx.deps, tool_name=tool_name, message=message, ok=ok)
    return message


@crowdin_toolset.tool
async def list_projects(  # noqa: RUF029
    ctx: RunContext[OrchestratorDeps],
) -> str:
    """List all Crowdin projects accessible with the configured token.

    Returns a JSON summary of project IDs and names.

    Returns:
        JSON string of project summaries.
    """
    try:
        items = crowdin_list_projects(
            token=_require_token(ctx.deps),
            organization=ctx.deps.crowdin_organization,
            base_url=ctx.deps.crowdin_base_url,
            timeout_seconds=ctx.deps.crowdin_timeout,
        )
        return _report(
            ctx,
            "crowdin_list_projects",
            summarize_list(items, label="projects"),
            True,
        )
    except Exception as e:  # noqa: BLE001
        return _report(
            ctx,
            "crowdin_list_projects",
            error_message("List Crowdin projects", e),
            False,
        )


@crowdin_toolset.tool
async def list_files(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    project: str,
) -> str:
    """List source files in a Crowdin project.

    Args:
        project: Crowdin project name or ID.

    Returns:
        JSON string of file summaries.
    """
    try:
        kwargs = _project_kwargs(project)
        items = crowdin_list_files(
            token=_require_token(ctx.deps),
            organization=ctx.deps.crowdin_organization,
            base_url=ctx.deps.crowdin_base_url,
            timeout_seconds=ctx.deps.crowdin_timeout,
            **kwargs,
        )
        return _report(
            ctx,
            "crowdin_list_files",
            summarize_list(items, label="files", name_field="path"),
            True,
        )
    except Exception as e:  # noqa: BLE001
        return _report(
            ctx,
            "crowdin_list_files",
            error_message("List Crowdin files", e),
            False,
        )


@crowdin_toolset.tool
async def list_languages(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    project: str,
) -> str:
    """List supported languages for a Crowdin project.

    Args:
        project: Crowdin project name or ID.

    Returns:
        JSON string of language summaries.
    """
    try:
        kwargs = _project_kwargs(project)
        items = crowdin_list_languages(
            token=_require_token(ctx.deps),
            organization=ctx.deps.crowdin_organization,
            base_url=ctx.deps.crowdin_base_url,
            timeout_seconds=ctx.deps.crowdin_timeout,
            **kwargs,
        )
        return _report(
            ctx,
            "crowdin_list_languages",
            summarize_list(items, label="languages", name_field="name"),
            True,
        )
    except Exception as e:  # noqa: BLE001
        return _report(
            ctx,
            "crowdin_list_languages",
            error_message("List Crowdin languages", e),
            False,
        )


@crowdin_toolset.tool
async def get_progress(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    project: str,
    language: str | None = None,
) -> str:
    """Get translation progress for a Crowdin project.

    Args:
        project: Crowdin project name or ID.
        language: Optional language code to filter by.

    Returns:
        JSON string of progress data.
    """
    try:
        kwargs = _project_kwargs(project)
        items = crowdin_progress(
            token=_require_token(ctx.deps),
            organization=ctx.deps.crowdin_organization,
            base_url=ctx.deps.crowdin_base_url,
            timeout_seconds=ctx.deps.crowdin_timeout,
            file_id=None,
            language=language,
            **kwargs,
        )
        return _report(
            ctx,
            "crowdin_get_progress",
            summarize_progress(items),
            True,
        )
    except Exception as e:  # noqa: BLE001
        return _report(
            ctx,
            "crowdin_get_progress",
            error_message("Get Crowdin progress", e),
            False,
        )


@crowdin_toolset.tool(requires_approval=True)
async def download_translation(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    project: str,
    file_id: int,
    language: str,
    output_path: str,
) -> str:
    """Download a translation file from Crowdin.

    Args:
        project: Crowdin project name or ID.
        file_id: Crowdin source file ID.
        language: Target language code.
        output_path: Local file path to save the download. Must end with
            ``.xliff`` or ``.xlf``.

    Returns:
        Confirmation message with output path.
    """
    try:
        kwargs = _project_kwargs(project)
        crowdin_download(
            token=_require_token(ctx.deps),
            organization=ctx.deps.crowdin_organization,
            base_url=ctx.deps.crowdin_base_url,
            timeout_seconds=ctx.deps.crowdin_timeout,
            file_id=file_id,
            language=language,
            output_path=output_path,
            **kwargs,
        )
        message = f"Downloaded to {output_path}"
        report_tool_outcome(
            ctx.deps,
            tool_name="crowdin_download_translation",
            message=message,
            ok=True,
        )
        return message
    except Exception as e:  # noqa: BLE001
        message = error_message("Crowdin download", e)
        report_tool_outcome(
            ctx.deps,
            tool_name="crowdin_download_translation",
            message=message,
            ok=False,
        )
        return message


@crowdin_toolset.tool(requires_approval=True)
async def upload_translation(  # noqa: RUF029, D417
    ctx: RunContext[OrchestratorDeps],
    project: str,
    file_id: int,
    language: str,
    file_path: str,
) -> str:
    """Upload a translation file to Crowdin.

    Args:
        project: Crowdin project name or ID.
        file_id: Crowdin source file ID.
        language: Target language code.
        file_path: Local translation file to upload. Must end with
            ``.xliff`` or ``.xlf``.

    Returns:
        Confirmation message.
    """
    try:
        kwargs = _project_kwargs(project)
        crowdin_upload(
            token=_require_token(ctx.deps),
            organization=ctx.deps.crowdin_organization,
            base_url=ctx.deps.crowdin_base_url,
            timeout_seconds=ctx.deps.crowdin_timeout,
            file_id=file_id,
            language=language,
            file_path=file_path,
            **kwargs,
        )
        message = f"Uploaded {file_path} to Crowdin"
        report_tool_outcome(
            ctx.deps,
            tool_name="crowdin_upload_translation",
            message=message,
            ok=True,
        )
        return message
    except Exception as e:  # noqa: BLE001
        message = error_message("Crowdin upload", e)
        report_tool_outcome(
            ctx.deps,
            tool_name="crowdin_upload_translation",
            message=message,
            ok=False,
        )
        return message
