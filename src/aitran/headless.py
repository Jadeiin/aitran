"""Headless adapters that run the same file agents used by the WebUI."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ToolReturnPart,
)

from aitran.agent_work import BatchProgress, ReviewPage, TranslationPage
from aitran.agents._base import build_model, fmt_base_url
from aitran.agents.reviewer import build_review_file_agent
from aitran.agents.translator import build_translation_file_agent
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import AgentStreamEvent

ProgressCallback = Callable[[str, BatchProgress], None]


def _translation_agent(
    model: str, api_key: str | None, api_host: str | None, temperature: float
):
    return build_translation_file_agent(
        build_model(
            model,
            api_key=api_key,
            base_url=fmt_base_url(api_host),
            temperature=temperature,
        )
    )


def _review_agent(
    model: str, api_key: str | None, api_host: str | None, temperature: float
):
    return build_review_file_agent(
        build_model(
            model,
            api_key=api_key,
            base_url=fmt_base_url(api_host),
            temperature=temperature,
        )
    )


def _run(
    agent,
    task: dict[str, Any],
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    progress_started = False

    async def handle_events(
        _ctx: RunContext[OrchestratorDeps],
        events: AsyncIterable[AgentStreamEvent],
    ) -> None:
        nonlocal progress_started
        async for event in events:
            if not isinstance(event, FunctionToolResultEvent):
                continue
            part = event.part
            if not isinstance(part, ToolReturnPart):
                continue
            if part.tool_name in {"get_translation_batch", "get_review_batch"}:
                if progress_started:
                    continue
                page_type = (
                    TranslationPage
                    if part.tool_name == "get_translation_batch"
                    else ReviewPage
                )
                page = page_type.model_validate(part.content)
                update = BatchProgress(
                    accepted=0,
                    total=page.total,
                    remaining=page.total,
                    complete=False,
                )
                progress_started = True
            elif part.tool_name in {
                "submit_translation_batch",
                "submit_review_batch",
            }:
                update = BatchProgress.model_validate(part.content)
            else:
                continue
            if on_progress is not None:
                on_progress(task["input_path"], update)

    result = agent.run_sync(
        json.dumps(task),
        deps=OrchestratorDeps(),
        event_stream_handler=handle_events if on_progress is not None else None,
    )
    return json.loads(result.output)


def translate_file(
    *,
    model: str,
    input_path: str,
    output_path: str,
    source_lang: str,
    target_lang: str,
    context_file: str | None,
    max_page_units: int,
    max_page_chars: int,
    scope: str = "pending",
    order: str = "file",
    profile: str = "full",
    instructions: str = "",
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one translation file through the translator Agent.

    Returns:
        Completed artifact payload.
    """
    agent = _translation_agent(model, api_key, api_host, temperature)
    return _run(
        agent,
        {
            "input_path": input_path,
            "output_path": output_path,
            "source_language": source_lang,
            "target_language": target_lang,
            "scope": scope,
            "context_file": context_file,
            "max_page_units": max_page_units,
            "max_page_chars": max_page_chars,
            "order": order,
            "profile": profile,
            "instructions": instructions,
        },
        on_progress=on_progress,
    )


def translate_directory(
    *,
    model: str,
    dir_path: str,
    suffixes: tuple[str, ...],
    source_lang: str,
    target_lang: str,
    context_file: str | None,
    max_page_units: int,
    max_page_chars: int,
    jobs: int,
    scope: str = "pending",
    order: str = "file",
    profile: str = "full",
    instructions: str = "",
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Run one isolated translator Agent per directory file.

    Returns:
        Artifact payloads in filename order.
    """
    paths = [
        os.path.join(dir_path, name)
        for name in sorted(os.listdir(dir_path))
        if name.endswith(suffixes)
    ]

    def run_path(path: str) -> dict[str, Any]:
        agent = _translation_agent(model, api_key, api_host, temperature)
        return _run(
            agent,
            {
                "input_path": path,
                "output_path": path,
                "source_language": source_lang,
                "target_language": target_lang,
                "scope": scope,
                "context_file": context_file,
                "max_page_units": max_page_units,
                "max_page_chars": max_page_chars,
                "order": order,
                "profile": profile,
                "instructions": instructions,
            },
            on_progress=on_progress,
        )

    with ThreadPoolExecutor(max_workers=min(jobs, len(paths)) or 1) as executor:
        return list(executor.map(run_path, paths))


def review_file(
    *,
    model: str,
    path: str,
    source_lang: str,
    target_lang: str,
    output_path: str,
    max_page_units: int,
    max_page_chars: int,
    api_key: str | None = None,
    api_host: str | None = None,
    temperature: float = 0.1,
    on_progress: ProgressCallback | None = None,
    instructions: str = "",
) -> dict[str, Any]:
    """Run one file through the full reviewer Agent.

    Returns:
        Completed review artifact payload.
    """
    agent = _review_agent(model, api_key, api_host, temperature)
    return _run(
        agent,
        {
            "input_path": path,
            "output_path": output_path,
            "source_language": source_lang,
            "target_language": target_lang,
            "max_page_units": max_page_units,
            "max_page_chars": max_page_chars,
            "instructions": instructions,
        },
        on_progress=on_progress,
    )
