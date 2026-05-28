"""Shared utilities for aitran agents.

Model routing, XML prompt builders, and language-label helpers used by
both the translator and reviewer agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.settings import ModelSettings
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential
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


def prompt_texts(value: object) -> list[str]:
    """Coerce singular or multistring text into XML-safe prompt strings.

    Returns:
        One XML-safe string per form.
    """
    strings = value.strings if hasattr(value, "strings") else [value]
    return [safe_prompt_text(s) for s in strings]


def build_unit_prompt_fields(
    unit,
    index: int,
    *,
    include_target: bool = False,
    plural_tags: list[str] | None = None,
    force_plural: bool = False,
) -> dict:
    """Build source/target prompt fields for a translation unit.

    Plural units use list fields (``sources`` and optionally ``targets``).
    Singular units use scalar fields (``source`` and optionally ``target``).

    Returns:
        Prompt-ready unit fields including ``index``.
    """
    has_plural = getattr(unit, "hasplural", lambda: False)()
    use_plural_fields = has_plural and (force_plural or plural_tags is not None)
    if use_plural_fields:
        fields: dict = {
            "index": index,
            "sources": prompt_texts(unit.source),
        }
        if include_target:
            fields["targets"] = prompt_texts(unit.target)
        return fields

    fields = {"index": index, "source": safe_prompt_text(unit.source)}
    if include_target:
        fields["target"] = safe_prompt_text(unit.target)
    return fields


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


def _raise_for_retryable_status(response: httpx.Response) -> None:
    if response.status_code in (408, 429) or response.status_code >= 500:
        response.raise_for_status()


_MODEL_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0)


def build_retrying_http_client(
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build an async HTTP client with provider-level transient retries.

    Args:
        transport: Optional base transport to wrap. Defaults to HTTPX's async
            transport.

    Returns:
        HTTPX async client using Pydantic AI's tenacity retry transport.
    """
    transport = AsyncTenacityTransport(
        RetryConfig(
            retry=retry_if_exception_type((
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
            )),
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=20),
                max_wait=300,
            ),
            stop=stop_after_attempt(5),
            reraise=True,
        ),
        wrapped=transport,
        validate_response=_raise_for_retryable_status,
    )
    return httpx.AsyncClient(transport=transport, timeout=_MODEL_HTTP_TIMEOUT)


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
        anthropic_kwargs = {"http_client": build_retrying_http_client()}
        if api_key is not None:
            anthropic_kwargs["api_key"] = api_key
        anthropic_provider = AnthropicProvider(
            **anthropic_kwargs,
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
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            http_client=build_retrying_http_client(),
        )
    else:
        provider_kwargs: dict = {"http_client": build_retrying_http_client()}
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
