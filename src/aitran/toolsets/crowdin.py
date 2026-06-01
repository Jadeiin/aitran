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


@crowdin_toolset.tool
async def list_projects(  # noqa: RUF029
    ctx: RunContext[OrchestratorDeps],
) -> str:
    """List all Crowdin projects accessible with the configured token.

    Returns a JSON summary of project IDs and names.

    Returns:
        JSON string of project summaries.
    """
    items = crowdin_list_projects(
        token=_require_token(ctx.deps),
        organization=ctx.deps.crowdin_organization,
        base_url=ctx.deps.crowdin_base_url,
        timeout_seconds=ctx.deps.crowdin_timeout,
    )
    return summarize_list(items, label="projects")


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
    kwargs = _project_kwargs(project)
    items = crowdin_list_files(
        token=_require_token(ctx.deps),
        organization=ctx.deps.crowdin_organization,
        base_url=ctx.deps.crowdin_base_url,
        timeout_seconds=ctx.deps.crowdin_timeout,
        **kwargs,
    )
    return summarize_list(items, label="files", name_field="path")


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
    kwargs = _project_kwargs(project)
    items = crowdin_list_languages(
        token=_require_token(ctx.deps),
        organization=ctx.deps.crowdin_organization,
        base_url=ctx.deps.crowdin_base_url,
        timeout_seconds=ctx.deps.crowdin_timeout,
        **kwargs,
    )
    return summarize_list(items, label="languages", name_field="name")


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
    return summarize_progress(items)


@crowdin_toolset.tool
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
    return f"Downloaded to {output_path}"


@crowdin_toolset.tool
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
    return f"Uploaded {file_path} to Crowdin"
