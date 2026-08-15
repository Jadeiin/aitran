"""Tests for the native Pydantic AI WebUI wiring."""

from __future__ import annotations

from aitran import app
from aitran.toolsets._base import OrchestratorDeps


class _FakeAgent:
    def __init__(self, calls):
        self.calls = calls

    def to_web(self, *, deps):
        self.calls.append(("to_web", deps))
        return "starlette-app"


def test_build_web_app_routes_models_and_deps(monkeypatch):
    calls = []
    deps = OrchestratorDeps(
        translate_model="anthropic:worker",
        translate_api_key="worker-key",
        translate_api_host="https://worker.example",
        translate_temperature=0.2,
        review_model="openai:reviewer",
        review_api_key="review-key",
        review_api_host="https://review.example",
        review_temperature=0.05,
    )

    def fake_build_model(spec, **kwargs):
        model = (spec, kwargs)
        calls.append(("build_model", model))
        return model

    def fake_build_orchestrator_agent(model, *, translator_model, reviewer_model):
        calls.append(("build_agent", model, translator_model, reviewer_model))
        return _FakeAgent(calls)

    monkeypatch.setattr(app, "build_model", fake_build_model)
    monkeypatch.setattr(app, "build_orchestrator_agent", fake_build_orchestrator_agent)

    result = app.build_web_app(
        orchestrator_model="openai:orchestrator",
        orchestrator_api_key="orchestrator-key",
        orchestrator_api_host="https://orchestrator.example",
        orchestrator_temperature=0.6,
        deps=deps,
    )

    assert result == "starlette-app"
    assert calls[0][1][0] == "openai:orchestrator"
    assert calls[0][1][1]["base_url"] == "https://orchestrator.example/v1"
    assert calls[1][1][0] == "anthropic:worker"
    assert calls[1][1][1]["base_url"] == "https://worker.example/v1"
    assert calls[2][1][0] == "openai:reviewer"
    assert calls[2][1][1]["base_url"] == "https://review.example/v1"
    assert calls[3][0] == "build_agent"
    assert calls[4] == ("to_web", deps)


def test_run_web_app_serves_native_app(monkeypatch):
    calls = []
    deps = OrchestratorDeps()

    monkeypatch.setattr(app, "build_web_app", lambda **_kwargs: "web-app")

    import uvicorn

    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda web_app, *, host, port: calls.append((web_app, host, port)),
    )

    app.run_web_app(
        orchestrator_model="openai:test",
        deps=deps,
        web_host="0.0.0.0",
        web_port=9000,
    )

    assert calls == [("web-app", "0.0.0.0", 9000)]
