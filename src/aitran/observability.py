"""Optional observability integrations for translation runs."""

from aitran.utils import aitran_version

_LOGFIRE_CONFIGURED = False
_LOGFIRE_HTTPX_INSTRUMENTED = False
_MLFLOW_CONFIGURED = False


class ObservabilityError(RuntimeError):
    """Raised when an observability backend cannot be configured."""


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
            service_version=aitran_version(),
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


def setup_mlflow(
    *,
    enabled: bool,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> bool:
    """Configure MLflow tracing for pydantic-ai.

    Args:
        enabled: Whether MLflow tracing should be configured.
        tracking_uri: Optional MLflow tracking server URI.
        experiment: Optional MLflow experiment name.

    Returns:
        True when MLflow was enabled, otherwise False.

    Raises:
        ObservabilityError: If MLflow is not installed.
    """
    global _MLFLOW_CONFIGURED

    if not enabled:
        return False

    try:
        import mlflow
    except ImportError as exc:
        raise ObservabilityError(
            "MLflow is not installed. Install it with `pip install mlflow>=3.1`."
        ) from exc

    if not _MLFLOW_CONFIGURED:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment:
            mlflow.set_experiment(experiment)
        mlflow.pydantic_ai.autolog()
        _MLFLOW_CONFIGURED = True

    return True


def flush_mlflow(*, enabled: bool) -> None:
    """Flush pending MLflow traces before CLI process exit."""
    if not enabled:
        return

    try:
        import mlflow
    except ImportError:
        return

    mlflow.flush_trace_async_logging()
