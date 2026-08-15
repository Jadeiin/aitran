"""Reviewer Agent that owns a complete paginated file task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.toolsets import DynamicToolset

from aitran.agent_work import ReviewResult as ReviewedUnit
from aitran.agent_work import review_toolset
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model

TASK_CONTRACT = """\
Review exactly one translated PO/XLIFF file. The task must be a JSON object:
{
  "input_path": string,          // required translated .po/.pot/.xliff/.xlf
  "output_path": string,         // required exact reviewed artifact destination
  "source_language": string,     // default "en"
  "target_language": string,     // required unless inferable from the file
  "max_page_units": integer > 0, // default 100
  "max_page_chars": integer > 0, // default 4096
  "instructions": string         // optional free-form review guidance
}
Put domain, tone, terminology, product, and other free-form requirements only in
`instructions`.
"""

SYSTEM_PROMPT = """\
You are a translation quality reviewer responsible for completing exactly one
JSON PO/XLIFF file task. Follow any task `instructions` and inspect every translation
returned by the tool, not only units with automated QA findings.

Workflow and output:
1. Retrieve the next bounded page with the single available retrieval tool.
2. Submit only problematic units with the exact `batch_id`. Omission means pass.
3. Use `revise` for an issue you can correct and always provide the complete final
   `corrected` target. Use `reject` for a serious issue requiring human/product
   judgment; provide a correction when safe, otherwise leave it null and explain.
4. Repeat until `finish_review` appears, call it, and return its artifact JSON
   verbatim. Never invent paths, batch IDs, or indices.

Review rules:
- Check meaning accuracy, completeness, naturalness, grammar, terminology and tone
  consistency, cultural appropriateness, and suitability for the supplied product
  context.
- Verify all placeholders, markup, escapes, accelerator markers, whitespace, line
  breaks, and punctuation against the source. Corrections must preserve functional
  formatting and must not XML-escape final text.
- Inspect every singular/plural source and target form in order. If a plural issue
  cannot be represented safely by one corrected target, reject it with an actionable
  note for human retranslation.
- Treat automated QA findings as evidence, not verdicts. Critical functional errors
  such as placeholder, printf, XML-tag, or escape mismatches are usually real.
  Cosmetic findings such as brackets, capitalization, or ending punctuation may be
  valid target-language choices. Omit a unit when the QA finding is a false positive.
- Use context, locations, notes, flags, current targets, and task instructions when
  deciding whether wording is intentional. Do not replace a correct translation just
  to express a stylistic preference.
- Notes must be brief, specific, and actionable. Explain the defect or required human
  decision; do not narrate routine passes.

Do not answer questions or provide a prose report. Perform the review workflow only.
"""


def build_review_file_agent(model: Model) -> Agent[OrchestratorDeps, str]:
    """Build the reviewer used by Harness and headless CLI.

    Returns:
        Complete-file reviewer Agent.
    """
    agent = Agent[OrchestratorDeps, str](
        model,
        name="reviewer",
        description=TASK_CONTRACT,
        deps_type=OrchestratorDeps,
        instructions=SYSTEM_PROMPT,
        toolsets=[DynamicToolset(review_toolset, per_run_step=False)],
    )

    @agent.output_validator
    def require_finished_file(ctx: RunContext[OrchestratorDeps], _output: str) -> str:
        for message in reversed(ctx.messages):
            for part in reversed(message.parts):
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_name == "finish_review"
                ):
                    return str(part.content)
        raise ModelRetry("Call finish_review before returning")

    return agent


build_reviewer_agent = build_review_file_agent

__all__ = [
    "SYSTEM_PROMPT",
    "TASK_CONTRACT",
    "ReviewedUnit",
    "build_review_file_agent",
    "build_reviewer_agent",
]
