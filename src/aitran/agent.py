"""Pydantic AI agent for batch translation.

Defines the structured output schema, dependencies, model routing, and the
translator agent itself. The agent treats each translation result like a human
translator would: it can mark a unit as fuzzy (uncertain) and leave a
translator-style note for downstream review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from pydantic_ai.models import Model

from aitran.prompts import load_system_prompt, load_user_prompt


def build_input_xml(units: list, start_index: int) -> str:
    """Format translation units as XML using Pydantic AI's `format_as_xml`.

    Only non-None context / comment keys are included, so the resulting XML
    has no `<context>null</context>` noise.

    Returns:
        XML string with a root `<translate-batch>` element.
    """
    items: list[dict] = []
    for i, u in enumerate(units):
        d: dict = {"index": start_index + i, "source": u.source}
        ctx = getattr(u, "context", None)
        if ctx:
            d["context"] = ctx
        comment = getattr(u, "comment", None)
        if comment:
            d["comment"] = comment
        items.append(d)

    return format_as_xml(items, root_tag="translate-batch", item_tag="translate")


class TranslatedUnit(BaseModel):
    """One translation result produced by the agent."""

    index: int = Field(description="Index matching the requested unit.")
    target: str = Field(description="Translated text.")
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


def build_model(
    model_spec: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
) -> Model:
    """Build a Pydantic AI model from a ``provider:model`` string.

    Provider routing delegates to pydantic-ai's built-in
    :func:`~pydantic_ai.providers.infer_provider_class`. Known providers
    (``deepseek``, ``groq``, ``mistral``, etc.) get their dedicated Provider
    class. Unknown providers fall back to ``OpenAIProvider`` for
    OpenAI-compatible gateways (supports custom *base_url*).

    Args:
        model_spec: ``"provider:model_name"`` string.
        api_key: Optional API key (provider-dependent env var is used when
            omitted).
        base_url: Custom base URL (only for ``openai`` / ``openai-chat``
            providers and unknown gateways).
        temperature: Model temperature (Anthropic also enables prompt caching).

    Returns:
        A configured Pydantic AI ``Model`` instance.

    Raises:
        ValueError: If *model_spec* does not contain a colon.
    """
    if ":" not in model_spec:
        raise ValueError(
            f"--model must be 'provider:model' (e.g. "
            f"'anthropic:claude-sonnet-4-5'); got {model_spec!r}"
        )
    provider_name, model_name = model_spec.split(":", 1)

    # Anthropic needs specialised model class + caching settings
    if provider_name == "anthropic":
        anthropic_provider = (
            AnthropicProvider(api_key=api_key) if api_key else AnthropicProvider()
        )
        return AnthropicModel(
            model_name,
            provider=anthropic_provider,
            settings=AnthropicModelSettings(
                temperature=temperature,
                anthropic_cache_instructions=True,
                anthropic_cache="5m",
            ),
        )

    # All other providers → OpenAIChatModel (accepts any OpenAI-compatible
    # provider).  Use pydantic-ai's built-in provider class dispatch.
    try:
        provider_cls = infer_provider_class(provider_name)
    except ValueError:
        # Unknown provider → OpenAI-compatible gateway with custom base_url
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    else:
        provider_kwargs: dict = {}
        if api_key is not None:
            provider_kwargs["api_key"] = api_key
        if provider_name in ("openai", "openai-chat") and base_url is not None:
            provider_kwargs["base_url"] = base_url
        provider = provider_cls(**provider_kwargs)

    return OpenAIChatModel(
        model_name,
        provider=provider,
        settings=ModelSettings(temperature=temperature),
    )


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
        deps_type=TranslationDeps,
        output_type=TranslationBatch,
        instructions=load_system_prompt() + "\n\n" + load_user_prompt(),
        retries={"output": 3},
    )

    @agent.instructions
    def task_and_glossary(ctx: RunContext[TranslationDeps]) -> str:
        parts = [
            f"Translate from `{ctx.deps.source_lang}` to `{ctx.deps.target_lang}` "
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
