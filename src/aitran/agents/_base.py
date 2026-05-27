"""Shared utilities for aitran agents.

Model routing, XML prompt builders, and language-label helpers used by
both the translator and reviewer agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import format_as_xml
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from translate.lang import data as lang_data
from translate.misc import xml_helpers

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def safe_prompt_text(value: object) -> str:
    """Coerce Toolkit strings to XML-safe prompt text.

    Returns:
        Text with characters invalid in XML removed.
    """
    return xml_helpers.valid_chars_only(str(value))


def format_language_label(code: str) -> str:
    """Format locale code with Translate Toolkit's language display name.

    Returns:
        Combined language code and display name.

    Raises:
        ValueError: If Translate Toolkit does not know the language code.
    """
    language = lang_data.get_language(code)
    if not language:
        raise ValueError(f"Unknown or ambiguous language code: {code!r}")
    name = language[0]
    return f"{code} - {name}"


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


def build_input_xml(units: list, start_index: int, *, profile: str = "full") -> str:
    """Format translation units as XML for the LLM prompt.

    Only non-empty fields are included per unit.  Metadata is read through
    :class:`~translate.storage.base.TranslationUnit` standard API methods
    so the same code works for PO and XLIFF.

    Args:
        units: Translation units.
        start_index: 1-based starting index for this batch.
        profile: ``"fast"`` includes only ``index`` + ``source``;
            ``"full"`` (default) adds ``context``, ``location``,
            ``note``, and ``flag``.

    Returns:
        XML string with a root ``<translate-batch>`` element.
    """
    items: list[dict] = []
    for i, u in enumerate(units):
        d: dict = {"index": start_index + i, "source": safe_prompt_text(u.source)}

        if profile == "fast":
            items.append(d)
            continue

        getctx = getattr(u, "getcontext", None)
        if callable(getctx):
            ctx = getctx()
            if ctx:
                d["context"] = safe_prompt_text(ctx)

        locs = getattr(u, "getlocations", None)
        if callable(locs) and locs():
            d["location"] = safe_prompt_text(", ".join(locs()))

        notes = getattr(u, "getnotes", None)
        if callable(notes):
            note_text = notes().strip()
            if note_text:
                d["note"] = safe_prompt_text(note_text)

        typecomments = getattr(u, "typecomments", None)
        if typecomments:
            if isinstance(typecomments, list):
                typecomments = ", ".join(typecomments)
            clean = (
                f.strip().removeprefix("#,").removeprefix("#").strip(" ,")
                for f in str(typecomments).split(",")
            )
            flags = [f for f in clean if f]
            if flags:
                d["flag"] = safe_prompt_text(", ".join(flags))

        items.append(d)

    return format_as_xml(items, root_tag="translate-batch", item_tag="translate")
