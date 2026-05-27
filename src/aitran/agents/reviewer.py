"""Reviewer agent — reviews translated PO/XLIFF units using QA context + LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry

from aitran.agents._base import format_language_label

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.models import Model

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a translation quality reviewer. You will evaluate existing "
    "translations against their source text and any QA rule violations "
    "detected by automated checkers. Your goal is to confirm real problems, "
    "dismiss false positives, and provide corrections when possible."
)

USER_PROMPT = """\
Review guidelines:

1. **Verdict** — For each unit, assign exactly one of:
   - `pass` — The translation is correct and natural.
   - `revise` — Minor issue found; provide a `corrected` target.
   - `reject` — Serious issue found. If you can fix it, provide `corrected`;
     if the problem requires human retranslation (e.g. the meaning is
     fundamentally wrong), leave `corrected` as null.

2. **Evaluating QA errors** — Automated checkers report violations with
   severity levels (critical / functional / cosmetic / extraction). Treat
   them as signals, not verdicts:
   - Critical errors (printf, xmltags, escapes) are almost always real.
   - Cosmetic errors (brackets, endpunc, caps) may be intentional style
     choices in the target language — use your judgment.
   - If a QA error is a false positive, still mark `pass` for that unit.

3. **Beyond QA errors** — Also check for:
   - Meaning accuracy: does the target convey the source intent?
   - Naturalness: does the target read like native text?
   - Consistency: are similar terms translated consistently?
   - Cultural appropriateness: any offensive or awkward phrasing?

4. **Notes** — Use the `note` field to explain your reasoning when the
   verdict is not `pass`. Keep notes brief and actionable. Do not narrate
   routine passes.

5. **Output format** — Return exactly one `ReviewedUnit` per requested
   index. Do not invent extra indices and do not omit any.

Do not answer questions or explain concepts. Review only."""


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class ReviewedUnit(BaseModel):
    """One review result produced by the agent."""

    index: int = Field(description="Index matching the requested unit.")
    verdict: str = Field(
        description=(
            "Review verdict: 'pass' (OK), 'revise' (minor issue, has "
            "correction), or 'reject' (serious issue)."
        ),
    )
    corrected: str | None = Field(
        default=None,
        description=(
            "Corrected target text. Present when verdict is 'revise' or "
            "'reject' and the reviewer can provide a fix. Null when the "
            "translation needs human retranslation."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Brief explanation of the verdict, especially for non-pass "
            "results. Helps human reviewers understand the issue."
        ),
    )


class ReviewBatch(BaseModel):
    """Container for a batch of review results."""

    units: list[ReviewedUnit]


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class ReviewDeps:
    """Per-run context injected into the reviewer agent at each batch."""

    source_lang: str
    target_lang: str
    context: str
    expected_indices: tuple[int, ...]


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------


def build_reviewer_agent(model: Model) -> Agent[ReviewDeps, ReviewBatch]:
    """Build the reviewer agent.

    Returns:
        A configured Pydantic AI Agent with output validation.
    """
    agent = Agent[ReviewDeps, ReviewBatch](
        model,
        name="aitran-reviewer",
        deps_type=ReviewDeps,
        output_type=ReviewBatch,
        instructions=SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
        retries={"output": 3},
    )

    @agent.instructions
    def task_context(ctx: RunContext[ReviewDeps]) -> str:
        source_label = format_language_label(ctx.deps.source_lang)
        target_label = format_language_label(ctx.deps.target_lang)
        parts = [
            f"Review translations from `{source_label}` to `{target_label}` "
            f"(XPG/POSIX locale names used in Unix-like systems and GNU Gettext)."
        ]
        if ctx.deps.context:
            parts.append(f"File context: {ctx.deps.context}")
        return "\n\n".join(parts)

    @agent.output_validator
    def check_completeness(
        ctx: RunContext[ReviewDeps], output: ReviewBatch
    ) -> ReviewBatch:
        if ctx.partial_output:
            return output
        got = {u.index for u in output.units}
        expected = set(ctx.deps.expected_indices)
        missing = expected - got
        extra = got - expected
        if missing or extra:
            msg_parts = []
            if missing:
                msg_parts.append(f"missing indices {sorted(missing)}")
            if extra:
                msg_parts.append(f"unexpected indices {sorted(extra)}")
            raise ModelRetry(
                "Review set incomplete: "
                + "; ".join(msg_parts)
                + ". Return exactly one entry per requested index."
            )
        # Validate verdict values
        valid_verdicts = {"pass", "revise", "reject"}
        for u in output.units:
            if u.verdict not in valid_verdicts:
                raise ModelRetry(
                    f"Invalid verdict {u.verdict!r} for index {u.index}. "
                    f"Must be one of: {', '.join(sorted(valid_verdicts))}"
                )
        return output

    return agent
