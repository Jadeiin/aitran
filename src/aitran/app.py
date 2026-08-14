"""Pydantic AI WebUI entry point for the aitran orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aitran.agents._base import build_model, fmt_base_url
from aitran.agents.orchestrator import build_orchestrator_agent
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from starlette.applications import Starlette


def build_web_app(
    *,
    orchestrator_model: str,
    orchestrator_api_key: str | None = None,
    orchestrator_api_host: str | None = None,
    orchestrator_temperature: float = 0.5,
    deps: OrchestratorDeps | None = None,
) -> Starlette:
    """Build the orchestrator's native Pydantic AI WebUI application.

    Returns:
        Starlette application created by :meth:`pydantic_ai.Agent.to_web`.
    """
    deps = deps or OrchestratorDeps()
    orchestrator = build_model(
        orchestrator_model,
        api_key=orchestrator_api_key,
        base_url=fmt_base_url(orchestrator_api_host),
        temperature=orchestrator_temperature,
    )
    worker = build_model(
        deps.translate_model,
        api_key=deps.translate_api_key,
        base_url=fmt_base_url(deps.translate_api_host),
        temperature=deps.translate_temperature,
    )
    agent = build_orchestrator_agent(orchestrator, worker_model=worker)
    return agent.to_web(deps=deps)


def run_web_app(
    *,
    orchestrator_model: str,
    orchestrator_api_key: str | None = None,
    orchestrator_api_host: str | None = None,
    orchestrator_temperature: float = 0.5,
    deps: OrchestratorDeps | None = None,
    web_host: str = "127.0.0.1",
    web_port: int = 7932,
) -> None:
    """Serve the orchestrator WebUI with Uvicorn."""
    import uvicorn

    web_app = build_web_app(
        orchestrator_model=orchestrator_model,
        orchestrator_api_key=orchestrator_api_key,
        orchestrator_api_host=orchestrator_api_host,
        orchestrator_temperature=orchestrator_temperature,
        deps=deps,
    )
    uvicorn.run(web_app, host=web_host, port=web_port)
