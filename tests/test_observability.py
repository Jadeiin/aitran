"""Tests for optional observability setup."""

from types import SimpleNamespace

from aitran import observability
from aitran.utils import aitran_version


def test_setup_logfire_disabled_does_nothing():
    assert observability.setup_logfire(enabled=False) is False


def test_setup_logfire_configures_pydantic_ai(monkeypatch):
    calls = []

    fake_logfire = SimpleNamespace(
        configure=lambda **kwargs: calls.append(("configure", kwargs)),
        instrument_pydantic_ai=lambda **kwargs: calls.append((
            "instrument_pydantic_ai",
            kwargs,
        )),
        instrument_httpx=lambda **kwargs: calls.append(("instrument_httpx", kwargs)),
        force_flush=lambda: calls.append(("force_flush", {})),
    )
    monkeypatch.setitem(__import__("sys").modules, "logfire", fake_logfire)
    monkeypatch.setattr(observability, "_LOGFIRE_CONFIGURED", False)
    monkeypatch.setattr(observability, "_LOGFIRE_HTTPX_INSTRUMENTED", False)

    assert observability.setup_logfire(enabled=True, capture_http=True) is True
    observability.flush_logfire(enabled=True)

    assert calls == [
        (
            "configure",
            {
                "service_name": "aitran",
                "service_version": aitran_version(),
                "send_to_logfire": "if-token-present",
                "console": False,
            },
        ),
        ("instrument_pydantic_ai", {"include_content": True}),
        ("instrument_httpx", {"capture_all": True}),
        ("force_flush", {}),
    ]


def test_setup_mlflow_disabled_does_nothing():
    assert observability.setup_mlflow(enabled=False) is False


def test_setup_mlflow_configures_pydantic_ai(monkeypatch):
    calls = []

    fake_pydantic_ai = SimpleNamespace(
        autolog=lambda: calls.append(("pydantic_ai.autolog", {})),
    )
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("set_tracking_uri", uri)),
        set_experiment=lambda name: calls.append(("set_experiment", name)),
        pydantic_ai=fake_pydantic_ai,
        flush_trace_async_logging=lambda: calls.append(("flush_trace_async_logging", {})),
    )
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(observability, "_MLFLOW_CONFIGURED", False)

    assert (
        observability.setup_mlflow(
            enabled=True,
            tracking_uri="http://localhost:5000",
            experiment="aitran",
        )
        is True
    )
    observability.flush_mlflow(enabled=True)

    assert calls == [
        ("set_tracking_uri", "http://localhost:5000"),
        ("set_experiment", "aitran"),
        ("pydantic_ai.autolog", {}),
        ("flush_trace_async_logging", {}),
    ]
