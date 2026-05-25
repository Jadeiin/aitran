"""Tests for optional observability setup."""

from types import SimpleNamespace

from aitran import observability


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
                "service_version": observability._service_version(),
                "send_to_logfire": "if-token-present",
                "console": False,
            },
        ),
        ("instrument_pydantic_ai", {"include_content": True}),
        ("instrument_httpx", {"capture_all": True}),
        ("force_flush", {}),
    ]
