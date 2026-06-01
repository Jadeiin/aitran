"""Orchestrator agent — coordinates translation workflows across platforms.

Unlike the translator and reviewer agents (which are pure prompt→structured-output),
the orchestrator uses tool calling to interact with Crowdin, Weblate, and the
translation/review engines.  It follows a "propose → confirm → execute" pattern
via pydantic-ai's Deferred Tools mechanism.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import (
    AgentCapability,
    HandleDeferredToolCalls,
    PrefixTools,
    Toolset,
)
from pydantic_ai_backends import ConsoleCapability
from pydantic_ai_backends.permissions import DEFAULT_RULESET
from pydantic_ai_toolguard import PermissionRule, ToolGuard
from translate.lang import data as lang_data

from aitran.toolsets._base import OrchestratorDeps
from aitran.toolsets._reporting import build_reporting_hooks

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai_toolguard.capability import (
        ApprovalCallback as ToolApprovalCallback,
    )

DeferredHandler = Callable[
    [RunContext[OrchestratorDeps], DeferredToolRequests],
    DeferredToolResults | Awaitable[DeferredToolResults | None] | None,
]

SYSTEM_PROMPT = (
    "You are a translation workflow orchestrator. You help users translate "
    "software projects hosted on localization platforms like Crowdin and "
    "Weblate."
)

USER_PROMPT = """\
Workflow guidelines:

1. **Understand the request**:
   - Parse the platform, project, target language, and any file/component scope.
   - If the user already gave a valid target language code, keep using it consistently.

2. **Gather information first**:
   - Use read-only tools (`crowdin_list_projects`, `crowdin_list_files`,
     `crowdin_list_languages`, `crowdin_get_progress`, `weblate_list_objects`,
     `weblate_get_stats`) to inspect the current state before proposing write actions.

3. **Propose before execution**:
   - Explain which files, language, and operations you plan to run.
   - Never call write tools without first proposing the plan.

4. **Execution order**:
   - Download the translation file.
   - Translate the file locally.
   - Review the translated file.
   - Upload the translated file back to the platform.

5. **Platform discipline**:
   - For Crowdin projects, use Crowdin tools.
   - For Weblate projects, use Weblate tools.
   - Do not mix platforms in one workflow unless the user explicitly asks.

6. **File format constraints**:
   - Downloaded translation files must use supported translation suffixes only:
     `.po`, `.xliff`, or `.xlf`.
   - `translate_file` and `review_translated_file` only support PO/XLIFF files or
     directories containing those files.
   - Do not download JSON, YAML, or other native source formats for this workflow.
   - For Weblate downloads, prefer `.po` when both PO and XLIFF are available because
     support is best.
   - When calling download tools, choose an `output_path` whose suffix matches the
     desired translation format so the backend can infer the correct export format.

7. **Language codes**:
   - Use Translate Toolkit language codes exactly when calling `translate_file` or
     `review_translated_file`.
   - Do not guess alternate spellings after approval if a valid toolkit code is
     already available from the user or platform metadata.
   - Supported Translate Toolkit codes: {supported_codes}

8. **User communication**:
   - If progress shows most strings are already translated, mention it and ask whether
     to proceed with only the untranslated strings.
   - Keep summaries concise and actionable.
   - If a tool returns an error, explain it clearly and suggest the next fix.
"""


def _build_orchestrator_system_prompt() -> str:
    """Build the orchestrator prompt with format and language-code guidance.

    Returns:
        Full system prompt text for the orchestrator agent.
    """
    supported_codes = ", ".join(sorted(lang_data.languages))
    return SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(supported_codes=supported_codes)


def build_orchestrator_agent(
    model: Model,
    *,
    deferred_handler: DeferredHandler | None = None,
    on_approval: ToolApprovalCallback | None = None,
    extra_capabilities: list[AgentCapability[OrchestratorDeps]] | None = None,
) -> Agent[OrchestratorDeps, str | DeferredToolRequests]:
    """Build the orchestrator agent with all toolsets.

    Args:
        model: Pydantic AI model instance for the orchestrator.
        deferred_handler: Handler for deferred tool calls (approval flow).
            When provided, the agent uses HandleDeferredToolCalls to resolve
            approvals inline, so the run returns a plain string.
        on_approval: Callback for ToolGuard approval-gated tools.
            Receives (tool_name, permission_rule), returns True to approve.
        extra_capabilities: Additional capabilities inserted before ToolGuard.
            Used for hooks that need to run before approval checks.

    Returns:
        Configured orchestrator agent.
    """
    from aitran.toolsets import crowdin_toolset, translate_toolset, weblate_toolset

    guard = ToolGuard(
        on_approval=on_approval,
        rules=[
            # Crowdin/Weblate write operations need approval
            PermissionRule(
                tool_pattern="crowdin_download_translation",
                action="allow",
                requires_approval=True,
            ),
            PermissionRule(
                tool_pattern="crowdin_upload_translation",
                action="allow",
                requires_approval=True,
            ),
            PermissionRule(
                tool_pattern="weblate_download_translation",
                action="allow",
                requires_approval=True,
            ),
            PermissionRule(
                tool_pattern="weblate_upload_translation",
                action="allow",
                requires_approval=True,
            ),
            # Translate/review need approval
            PermissionRule(
                tool_pattern="translate_file",
                action="allow",
                requires_approval=True,
            ),
            PermissionRule(
                tool_pattern="review_translated_file",
                action="allow",
                requires_approval=True,
            ),
            # Read-only tools — allow without approval
            PermissionRule(tool_pattern="crowdin_*", action="allow"),
            PermissionRule(tool_pattern="weblate_*", action="allow"),
            # Console/file tools — ToolGuard allows them through;
            # ConsoleCapability enforces per-path permissions (secrets deny, write/edit ask)
            PermissionRule(tool_pattern="read_file", action="allow"),
            PermissionRule(tool_pattern="write_file", action="allow"),
            PermissionRule(tool_pattern="edit_file", action="allow"),
            PermissionRule(tool_pattern="ls", action="allow"),
            PermissionRule(tool_pattern="glob", action="allow"),
            PermissionRule(tool_pattern="grep", action="allow"),
            # No wildcard deny rule needed — ToolGuard's default is deny.
        ],
    )

    reporting_hooks = build_reporting_hooks()
    capabilities: list[AgentCapability[OrchestratorDeps]] = [
        PrefixTools(wrapped=Toolset(crowdin_toolset), prefix="crowdin"),
        PrefixTools(wrapped=Toolset(weblate_toolset), prefix="weblate"),
        Toolset(translate_toolset),
        *(extra_capabilities or []),
        guard,
        ConsoleCapability(permissions=DEFAULT_RULESET, include_execute=False),
        reporting_hooks,  # type: ignore[arg-type]
    ]
    if deferred_handler is not None:
        capabilities.append(
            HandleDeferredToolCalls[OrchestratorDeps](handler=deferred_handler)
        )

    return Agent(
        model,
        deps_type=OrchestratorDeps,
        output_type=[str, DeferredToolRequests],
        system_prompt=_build_orchestrator_system_prompt(),
        capabilities=capabilities,
    )
