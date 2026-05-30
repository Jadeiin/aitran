"""Tests for orchestrator Weblate toolset wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from aitran.toolsets._base import OrchestratorDeps
from aitran.toolsets.weblate import get_stats


async def test_get_stats_reports_result_to_terminal(monkeypatch):
    reports: list[tuple[str, str, bool]] = []

    monkeypatch.setattr(
        "aitran.toolsets.weblate.weblate_stats",
        lambda **_kwargs: {"total": 10, "translated": 5},
    )

    ctx = SimpleNamespace(
        deps=OrchestratorDeps(
            weblate_url="https://example.com",
            weblate_token="token",
            tool_reporter=lambda *args: reports.append(args),
        )
    )

    result = await get_stats(ctx, "project/component/zh_Hans")

    assert '"total": 10' in result
    assert reports == [("weblate__get_stats", result, True)]
