"""Orchestrator agent — coordinates translation workflows across platforms.

Unlike the translator and reviewer agents (which are pure prompt→structured-output),
the orchestrator uses tool calling to interact with Crowdin, Weblate, and the
translation/review engines.  It follows a "propose → confirm → execute" pattern
via pydantic-ai's Deferred Tools mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls

from aitran.agents._base import build_model
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model

DeferredHandler = Callable[
    [RunContext[OrchestratorDeps], DeferredToolRequests],
    DeferredToolResults,
]

SYSTEM_PROMPT = """\
You are a translation workflow orchestrator. Your job is to help users translate \
software projects hosted on localization platforms like Crowdin and Weblate.

## How you work

1. **Understand the user's intent** — parse the platform, project, target language, \
and any file/language scope from their natural-language request.
2. **Gather information** — use read-only tools (crowdin_list_projects, \
crowdin_list_files, crowdin_list_languages, crowdin_get_progress, \
weblate_list_objects, weblate_get_stats) to understand the current state.
3. **Propose a plan** — tell the user what you intend to do (which files, which \
languages, what operations).  Be specific and concise.
4. **Execute** — call the write tools (crowdin_download_translation, \
weblate_download_translation, translate_file, crowdin_upload_translation, \
weblate_upload_translation, review_translated_file).  These require approval.
5. **Report** — summarize what was done, any errors, and what remains.

## Rules

- Always propose before executing. Never call write tools without first \
explaining what you will do.
- For Crowdin projects, use the Crowdin tools. For Weblate projects, use the \
Weblate tools. Never mix platforms in a single workflow unless the user asks.
- When translating, download the file first, translate locally, then upload the \
translated file back.
- If translation progress shows most strings are already translated, mention \
this and ask if the user wants to proceed with only the untranslated strings.
- Keep summaries concise. Focus on actionable information.
- If a tool returns an error, explain it clearly and suggest a fix.
"""


def build_orchestrator_agent(
    model: Model,
    *,
    deferred_handler: DeferredHandler | None = None,
) -> Agent[OrchestratorDeps, str | DeferredToolRequests]:
    """Build the orchestrator agent with all toolsets.

    Args:
        model: Pydantic AI model instance for the orchestrator.
        deferred_handler: Handler for deferred tool calls (approval flow).
            When provided, the agent uses HandleDeferredToolCalls to resolve
            approvals inline, so the run returns a plain string.

    Returns:
        Configured orchestrator agent.
    """
    from pydantic_ai.toolsets import PrefixedToolset

    from aitran.toolsets import crowdin_toolset, translate_toolset, weblate_toolset

    capabilities: list = []
    if deferred_handler is not None:
        capabilities.append(
            HandleDeferredToolCalls[OrchestratorDeps](handler=deferred_handler)
        )

    return Agent(
        model,
        deps_type=OrchestratorDeps,
        output_type=[str, DeferredToolRequests],
        system_prompt=SYSTEM_PROMPT,
        toolsets=[
            PrefixedToolset(crowdin_toolset, "crowdin_"),
            PrefixedToolset(weblate_toolset, "weblate_"),
            translate_toolset,
        ],
        capabilities=capabilities,
    )


def build_orchestrator_model(
    model_spec: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Model:
    """Build the model for the orchestrator agent.

    Falls back to a sensible default if no model spec is given.

    Args:
        model_spec: ``"provider:model"`` string, or None for default.
        api_key: Optional API key override.
        base_url: Optional base URL override.

    Returns:
        Configured model instance.
    """
    spec = model_spec or "anthropic:claude-sonnet-4-6"
    return build_model(spec, api_key=api_key, base_url=base_url, temperature=0.1)
