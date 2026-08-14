"""Orchestrator agent — coordinates translation workflows across platforms.

The orchestrator uses tool calling to interact with Crowdin and Weblate and to
delegate file work to the tool-using translator and reviewer subagents. Write
operations use Pydantic AI's native deferred-tool approval mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import PrefixTools, Toolset
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from translate.lang import data as lang_data

from aitran.agents.workflows import (
    build_review_workflow_agent,
    build_translation_workflow_agent,
)
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model

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
   - Treat file paths as a strict pipeline: use each tool/subagent result's
     `output_path` as the next step's `input_file` or upload `file_path`.
   - Never infer the next path from a naming convention and never upload an earlier
     pipeline input when a later `output_path` was returned.
   - For Weblate uploads, use `conflicts=replace-translated` when the translated or
     reviewed file corrects existing translations; keep `ignore` only when filling
     previously untranslated strings.

5. **Platform discipline**:
   - For Crowdin projects, use Crowdin tools.
   - For Weblate projects, use Weblate tools.
   - Do not mix platforms in one workflow unless the user explicitly asks.

6. **File format constraints**:
   - Downloaded translation files must use supported translation suffixes only:
     `.po`, `.xliff`, or `.xlf`.
   - Delegate local file work with `delegate_task`: choose `translator` for
     translation and `reviewer` for review. Give the subagent a self-contained JSON
     task containing all arguments it needs.
   - Translator task keys are `input_file`, `output_file`, `source_language`,
     `target_language`, `context_file`, `batch_size`, `jobs`, `order`, and `profile`.
   - Reviewer task keys are `input_file`, `output_file`, `source_language`,
     `target_language`, `batch_size`, `strict`, and `auto_fix`.
   - The translator and reviewer only support PO/XLIFF files or directories
     containing those files.
   - Do not download JSON, YAML, or other native source formats for this workflow.
   - For Weblate downloads, prefer `.po` when both PO and XLIFF are available because
     support is best.
   - When calling download tools, choose an `output_path` whose suffix matches the
     desired translation format so the backend can infer the correct export format.

7. **Language codes**:
   - Use Translate Toolkit language codes exactly in translator and reviewer tasks.
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


class _ApprovalRequiredSubAgents(SubAgents[OrchestratorDeps]):
    """Expose Harness delegation through native deferred-tool approval."""

    def get_toolset(self):
        """Wrap the Harness delegate tool so the WebUI approves each workflow.

        Returns:
            Approval-gated delegate toolset, or none when no subagents exist.
        """
        toolset = super().get_toolset()
        return toolset.approval_required() if toolset is not None else None


def build_orchestrator_agent(
    model: Model,
    *,
    worker_model: Model | None = None,
) -> Agent[OrchestratorDeps, str | DeferredToolRequests]:
    """Build the orchestrator agent with all toolsets.

    Args:
        model: Pydantic AI model instance for the orchestrator.
        worker_model: Model used by translation and review workflow subagents.
            Defaults to the orchestrator model.

    Returns:
        Configured orchestrator agent.
    """
    from aitran.toolsets import crowdin_toolset, weblate_toolset

    worker_model = worker_model or model
    translator = build_translation_workflow_agent(worker_model)
    reviewer = build_review_workflow_agent(worker_model)

    subagents = _ApprovalRequiredSubAgents(
        agents=[SubAgent(translator), SubAgent(reviewer)],
        agent_folders=None,
        forward_usage=True,
        inherit_tools=False,
        contain_errors=True,
    )

    capabilities = [
        PrefixTools(wrapped=Toolset(crowdin_toolset), prefix="crowdin"),
        PrefixTools(wrapped=Toolset(weblate_toolset), prefix="weblate"),
        subagents,
    ]

    return Agent(
        model,
        deps_type=OrchestratorDeps,
        output_type=[str, DeferredToolRequests],
        system_prompt=_build_orchestrator_system_prompt(),
        capabilities=capabilities,
    )
