"""Tests for orchestrator Weblate toolset wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from aitran.toolsets._base import OrchestratorDeps
from aitran.toolsets.weblate import get_stats, upload_translation


async def test_get_stats(monkeypatch):
    monkeypatch.setattr(
        "aitran.toolsets.weblate.weblate_stats",
        lambda **_kwargs: {"total": 10, "translated": 5},
    )

    ctx = SimpleNamespace(
        deps=OrchestratorDeps(
            weblate_url="https://example.com",
            weblate_token="token",
        )
    )

    result = await get_stats(ctx, "project/component/zh_Hans")

    assert '"total": 10' in result


async def test_upload_returns_weblate_counts(monkeypatch):
    captured = {}

    def fake_upload(**kwargs):
        captured.update(kwargs)
        return {"count": 4, "skipped": 1}

    monkeypatch.setattr("aitran.toolsets.weblate.weblate_upload", fake_upload)
    ctx = SimpleNamespace(
        deps=OrchestratorDeps(
            weblate_url="https://example.com",
            weblate_token="token",
        )
    )

    result = await upload_translation(
        ctx,
        "project/component/zh_Hans",
        "messages.reviewed.po",
        conflicts="replace-translated",
    )

    assert captured["file_path"] == "messages.reviewed.po"
    assert captured["conflicts"] == "replace-translated"
    assert '"skipped": 1' in result
