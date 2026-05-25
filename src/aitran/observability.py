"""Optional observability integrations for translation runs."""

from importlib.metadata import PackageNotFoundError, version

_LOGFIRE_CONFIGURED = False
_LOGFIRE_HTTPX_INSTRUMENTED = False


class ObservabilityError(RuntimeError):
    """Raised when an observability backend cannot be configured."""


def _service_version() -> str:
    try:
        return version("aitran")
    except PackageNotFoundError:
        return "unknown"


def setup_logfire(*, enabled: bool, capture_http: bool = False) -> bool:
    """Configure Pydantic Logfire instrumentation.

    Args:
        enabled: Whether Logfire should be configured.
        capture_http: Whether to capture raw HTTP request/response data.

    Returns:
        True when Logfire was enabled, otherwise False.

    Raises:
        ObservabilityError: If Logfire support is not installed.
    """
    global _LOGFIRE_CONFIGURED, _LOGFIRE_HTTPX_INSTRUMENTED

    if not enabled:
        return False

    try:
        import logfire
    except ImportError as exc:
        raise ObservabilityError(
            "Logfire support is not installed. Install the logfire extra, e.g. "
            "`uv sync` with pydantic-ai-slim[logfire] available."
        ) from exc

    if not _LOGFIRE_CONFIGURED:
        logfire.configure(
            service_name="aitran",
            service_version=_service_version(),
            send_to_logfire="if-token-present",
            console=False,
        )
        logfire.instrument_pydantic_ai(include_content=True)
        _LOGFIRE_CONFIGURED = True

    if capture_http and not _LOGFIRE_HTTPX_INSTRUMENTED:
        logfire.instrument_httpx(capture_all=True)
        _LOGFIRE_HTTPX_INSTRUMENTED = True

    return True


def flush_logfire(*, enabled: bool) -> None:
    """Flush pending Logfire spans before CLI process exit."""
    if not enabled:
        return

    try:
        import logfire
    except ImportError:
        return

    logfire.force_flush()
