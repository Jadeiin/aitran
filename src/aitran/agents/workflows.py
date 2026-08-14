"""Specialized workflow subagents delegated to by the orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from aitran.toolsets._base import OrchestratorDeps
from aitran.toolsets.translate import review_translated_file, translate_file

if TYPE_CHECKING:
    from pydantic_ai.models import Model

TaskT = TypeVar("TaskT", bound=BaseModel)


class TranslationWorkflowTask(BaseModel):
    """Deterministic translation task parsed from Harness delegation text."""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(validation_alias=AliasChoices("path", "input_file"))
    source_lang: str = Field(
        default="en", validation_alias=AliasChoices("source_lang", "source_language")
    )
    target_lang: str = Field(
        default="", validation_alias=AliasChoices("target_lang", "target_language")
    )
    output_path: str | None = Field(
        default=None, validation_alias=AliasChoices("output_path", "output_file")
    )
    context_file: str | None = None
    batch_size: int = 100
    jobs: int = 4
    order: str = "file"
    profile: str = "full"


class ReviewWorkflowTask(BaseModel):
    """Deterministic review task parsed from Harness delegation text."""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(validation_alias=AliasChoices("path", "input_file"))
    source_lang: str = Field(
        default="en", validation_alias=AliasChoices("source_lang", "source_language")
    )
    target_lang: str = Field(
        default="", validation_alias=AliasChoices("target_lang", "target_language")
    )
    output_path: str | None = Field(
        default=None, validation_alias=AliasChoices("output_path", "output_file")
    )
    batch_size: int = 100
    strict: bool = False
    auto_fix: bool = False


def _delegated_task(ctx: RunContext[OrchestratorDeps], task_type: type[TaskT]) -> TaskT:
    """Parse the original Harness task instead of letting the model copy arguments.

    Returns:
        Validated workflow task.

    Raises:
        ModelRetry: If the delegation is missing a JSON task.
    """
    for message in reversed(ctx.messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                try:
                    return task_type.model_validate_json(part.content)
                except ValidationError as exc:
                    raise ModelRetry(f"Invalid delegated task JSON: {exc}") from exc
    raise ModelRetry("The delegation must contain a self-contained JSON task.")


def _required_tool_result(ctx: RunContext[OrchestratorDeps], tool_name: str) -> str:
    """Return the workflow tool result verbatim.

    Returns:
        Exact tool result produced by the deterministic file worker.

    Raises:
        ModelRetry: If the workflow tool has not run successfully.
    """
    for message in reversed(ctx.messages):
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                return str(part.content)
    raise ModelRetry(f"Call {tool_name} before returning a result.")


def build_translation_workflow_agent(model: Model) -> Agent[OrchestratorDeps, str]:
    """Build the subagent that owns one translation-file delegation.

    Returns:
        Configured translation workflow subagent.
    """
    agent = Agent[OrchestratorDeps, str](
        model,
        name="translator",
        description="Translates PO/XLIFF files using the aitran translation pipeline.",
        deps_type=OrchestratorDeps,
        instructions=(
            "You are the translation workflow subagent. The task is a JSON object. "
            "Call the no-argument `run_translation` tool exactly once; it parses the "
            "original task deterministically. Then return its result verbatim. "
            "Do not inspect or modify files through any other mechanism."
        ),
    )

    @agent.tool
    async def run_translation(ctx: RunContext[OrchestratorDeps]) -> str:
        """Translate the assigned PO/XLIFF path using the language subagent.

        Returns:
            Translation workflow result.
        """
        task = _delegated_task(ctx, TranslationWorkflowTask)
        return await translate_file(ctx, **task.model_dump())

    @agent.output_validator
    def require_translation_tool(
        ctx: RunContext[OrchestratorDeps], _output: str
    ) -> str:
        return _required_tool_result(ctx, "run_translation")

    return agent


def build_review_workflow_agent(model: Model) -> Agent[OrchestratorDeps, str]:
    """Build the subagent that owns one translated-file review delegation.

    Returns:
        Configured review workflow subagent.
    """
    agent = Agent[OrchestratorDeps, str](
        model,
        name="reviewer",
        description="Reviews and optionally fixes translated PO/XLIFF files.",
        deps_type=OrchestratorDeps,
        instructions=(
            "You are the review workflow subagent. The task is a JSON object "
            "Call the no-argument `run_review` tool exactly once; it parses the "
            "original task deterministically. Then return its result verbatim. "
            "Do not inspect or modify files through any other mechanism."
        ),
    )

    @agent.tool
    async def run_review(ctx: RunContext[OrchestratorDeps]) -> str:
        """Review the assigned translated-file path.

        Returns:
            Review workflow result.
        """
        task = _delegated_task(ctx, ReviewWorkflowTask)
        return await review_translated_file(ctx, **task.model_dump())

    @agent.output_validator
    def require_review_tool(ctx: RunContext[OrchestratorDeps], _output: str) -> str:
        return _required_tool_result(ctx, "run_review")

    return agent
