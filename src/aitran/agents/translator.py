"""Translator Agent that owns a complete paginated file task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.toolsets import DynamicToolset

from aitran.agent_work import TranslationResult as TranslatedUnit
from aitran.agent_work import translation_toolset
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model

TASK_CONTRACT = """\
Translate exactly one PO/XLIFF file. The task must be a JSON object with this schema:
{
  "input_path": string,                 // required existing .po/.pot/.xliff/.xlf
  "output_path": string,                // required exact artifact destination
  "source_language": string,            // default "en"
  "target_language": string,            // required unless inferable from the file
  "scope": "pending" | "repair" | "all", // default "pending"
  "context_file": string | null,         // optional path to additional context
  "max_page_units": integer > 0,         // default 100
  "max_page_chars": integer > 0,         // default 4096
  "order": "file" | "source" | "reference" | "context", // default "file"
  "profile": "fast" | "full",           // metadata detail only; default "full"
  "instructions": string                 // optional free-form translation guidance
}
`profile` never contains prose. Put domain, tone, terminology, product, and other
free-form requirements in `instructions`.
"""

SYSTEM_PROMPT = """\
You are a language translation expert responsible for completing exactly one
JSON PO/XLIFF file task. Follow the task's `instructions` as translation guidance;
`profile` only controls how much unit metadata the retrieval tool returns.

Workflow:
1. Call the single available retrieval tool to get the next bounded page.
2. Translate every returned unit, then call `submit_translation_batch` once with
   the exact `batch_id` and exactly one result for every returned index.
3. Repeat until `finish_translation` becomes available. Call it and return its
   artifact JSON verbatim. Never invent paths, batch IDs, or unit indices.

Translation rules:
- Preserve every placeholder and its semantics, including printf forms (`%s`,
  `%1$d`), Python/brace forms (`{name}`), ICU variables, escapes, and accelerator
  markers. Never translate placeholders or silently add/remove them.
- Preserve leading and trailing whitespace, line breaks, intentional punctuation,
  capitalization constraints, and other untranslatable formatting. Match terminal
  punctuation unless target-language convention or explicit instructions require
  otherwise.
- Preserve markup structure and literal entities. Do not XML-escape returned text:
  return `<`, `>`, `&`, and quotes as the final file should contain them. Preserve
  an entity such as `&lt;` only when it was literal source content.
- Use `context`, `location`, developer/translator `note`, flags, current `targets`,
  file context, and task instructions to disambiguate. Flags such as `c-format`,
  `python-format`, and `fuzzy` constrain the translation.
- Apply glossary translations exactly when their source term occurs, while making
  the surrounding sentence natural and consistent.
- `sources` and `targets` are lists. Singular units require one target. For a
  plural unit, preserve plural-form order and return exactly the number of forms
  specified by `plural_tags`; do not merge or reorder forms.
- Existing targets are reference material for fuzzy, repair, and all-scope work.
  Return the complete final target for every selected unit, not a patch or comment.
- Set `fuzzy=true` only for genuine ambiguity, insufficient context, uncertain
  placeholders, or content requiring human attention. Do not use it for routine
  translations.
- Add a brief translator note only when a human reviewer needs an ambiguity,
  alternative wording, product decision, or context check. Do not narrate routine
  work.

Do not answer questions or explain the translations. Perform the tool workflow only.
"""


def build_translation_file_agent(model: Model) -> Agent[OrchestratorDeps, str]:
    """Build the translator used by Harness and headless CLI.

    Returns:
        Complete-file translator Agent.
    """
    agent = Agent[OrchestratorDeps, str](
        model,
        name="translator",
        description=TASK_CONTRACT,
        deps_type=OrchestratorDeps,
        instructions=SYSTEM_PROMPT,
        toolsets=[DynamicToolset(translation_toolset, per_run_step=False)],
    )

    @agent.output_validator
    def require_finished_file(ctx: RunContext[OrchestratorDeps], _output: str) -> str:
        for message in reversed(ctx.messages):
            for part in reversed(message.parts):
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_name == "finish_translation"
                ):
                    return str(part.content)
        raise ModelRetry("Call finish_translation before returning")

    return agent


# The concise name is the public builder; the longer alias documents its role.
build_translator_agent = build_translation_file_agent

__all__ = [
    "SYSTEM_PROMPT",
    "TASK_CONTRACT",
    "TranslatedUnit",
    "build_translation_file_agent",
    "build_translator_agent",
]
