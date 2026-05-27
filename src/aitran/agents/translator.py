"""Translator agent — translates batches of PO/XLIFF units via LLM."""

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
    "You are a language translation expert. "
    "You will carefully follow the translation guidelines "
    "to translate the incoming XML messages from one language to another."
)

USER_PROMPT = """\
Translation guidelines are as follows:

1. **Placeholder Handling**:
   - Maintain the positions of placeholders (e.g., %s, %d, {example}) in the translated text. Do not translate placeholders.

2. **Formatting**:
   - Preserve the formatting of untranslatable portions.
   - Retain any whitespace at the beginning or end of the message.
   - Add or omit a period (.) at the end of your translation to match the incoming message.

3. **Input Format**:
   - Messages arrive as `<translate-batch>` containing one `<translate>` element per unit.
   - Each `<translate>` has `<index>`, `<source>`, and optionally:
     * `<context>` — disambiguation context (PO `msgctxt`, XLIFF `context-group`).  Use this to distinguish homograph strings.
     * `<location>` — source-code references (e.g. `src/ui/mainwindow.cpp:42`).  Use this to infer the domain and intent of the string.
     * `<note>` — human annotations: developer comments, prior translator remarks, and tool diagnostics combined.
     * `<flag>` — format / state flags (`c-format`, `python-format`, `fuzzy`, etc.).  These constrain how placeholders must be handled.
   - Example:
     ```
     <translate-batch>
       <translate>
         <index>1</index>
         <source>File</source>
         <context>Menu</context>
         <location>src/ui/mainwindow.cpp:42</location>
         <note>Appears in the menu bar</note>
         <flag>c-format</flag>
       </translate>
       <translate>
         <index>2</index>
         <source>Hello %s</source>
       </translate>
     </translate-batch>
     ```

4. **Output Format**:
   - For each `<translate>` element you receive, produce exactly one `TranslatedUnit` with a matching `index`.
   - `targets` holds your translation(s) as a list. Single-element list for singular units; multi-element list for plural units matching the number of plural forms. Return the final text exactly as it should be saved. Do not XML-escape characters — write `"` not `&quot;`, `<` not `&lt;`, `>` not `&gt;`, `&` not `&amp;`. Only preserve XML entities that were literal in the source (e.g. if the source contains the literal string `&lt;code&gt;`, keep it as `&lt;code&gt;`).
   - `fuzzy` (default false): set to `true` when you are not confident — the source is ambiguous, placeholders are unclear, context is insufficient, or the string seems untranslatable. A reviewer will be alerted.
   - `note` (optional): leave a short translator-style remark only when it would help a human reviewer — alternative renderings, ambiguities, or context to verify. Keep notes brief; do not narrate routine translations.

5. **Multiple Translations**:
   - You may receive multiple translation units in a single `<translate-batch>`.
   - Return exactly one `TranslatedUnit` per requested index. Do not invent extra indices and do not omit any.

6. **Glossary**:
   - If a glossary is provided, use the listed translation whenever the source string contains the key (case-insensitive substring). Do not paraphrase glossary entries.

Do not answer questions or explain concepts. Translate only."""


class TranslatedUnit(BaseModel):
    """One translation result produced by the agent."""

    index: int = Field(description="Index matching the requested unit.")
    targets: list[str] = Field(
        description=(
            "Translated text(s). Single-element list for singular units; "
            "multi-element list for plural units matching plural_forms count."
        ),
    )
    fuzzy: bool = Field(
        default=False,
        description=(
            "Set true when the translation is uncertain — ambiguous "
            "placeholders, missing context, multiple plausible renderings, "
            "or untranslatable source. Maps to PO `#, fuzzy` flag / XLIFF "
            "state='needs-review-translation'."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Optional translator-style comment for the reviewer: alternative "
            "renderings, ambiguity flags, or context worth double-checking. "
            "Appended to PO translator comments / XLIFF <note from='translator'>."
        ),
    )


class TranslationBatch(BaseModel):
    """Container for a batch of translations."""

    translations: list[TranslatedUnit]


@dataclass
class TranslationDeps:
    """Per-run context injected into the translator agent at each batch."""

    source_lang: str
    target_lang: str
    context: str
    dict_entries: list[tuple[str, str]]
    expected_indices: tuple[int, ...]
    plural_tags: list[str] | None = None


def build_translator_agent(model: Model) -> Agent[TranslationDeps, TranslationBatch]:
    """Build the translator agent.

    Static `instructions` (system + user prompt) sit before any dynamic
    instructions, which lets Anthropic auto-place the cache breakpoint at the
    end of the stable prefix. The dynamic instruction injects per-run task
    description, file-level context, and matched glossary entries.

    Returns:
        A configured Pydantic AI Agent with output validation.
    """
    agent = Agent[TranslationDeps, TranslationBatch](
        model,
        name="aitran-translator",
        deps_type=TranslationDeps,
        output_type=TranslationBatch,
        instructions=SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
        retries={"output": 3},
    )

    @agent.instructions
    def task_and_glossary(ctx: RunContext[TranslationDeps]) -> str:
        source_label = format_language_label(ctx.deps.source_lang)
        target_label = format_language_label(ctx.deps.target_lang)
        parts = [
            f"Translate from `{source_label}` to `{target_label}` "
            f"(XPG/POSIX locale names used in Unix-like systems and GNU Gettext)."
        ]
        if ctx.deps.context:
            parts.append(f"File context: {ctx.deps.context}")
        if ctx.deps.dict_entries:
            lines = [f"- {k} → {v}" for k, v in ctx.deps.dict_entries]
            parts.append(
                "Glossary (use these translations exactly when the source "
                "string contains the key):\n" + "\n".join(lines)
            )
        if ctx.deps.plural_tags:
            tags = ", ".join(ctx.deps.plural_tags)
            parts.append(
                f"Target language has {len(ctx.deps.plural_tags)} plural "
                f"forms (in order): [{tags}]. "
                f"For plural units, provide exactly this many targets."
            )
        return "\n\n".join(parts)

    @agent.output_validator
    def check_completeness(
        ctx: RunContext[TranslationDeps], output: TranslationBatch
    ) -> TranslationBatch:
        if ctx.partial_output:
            return output
        got = {u.index for u in output.translations}
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
                "Translation set incomplete: "
                + "; ".join(msg_parts)
                + ". Return exactly one entry per requested index."
            )
        return output

    return agent
