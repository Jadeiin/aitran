"""Tests for orchestrator translate toolset wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from aitran.toolsets._base import OrchestratorDeps
from aitran.toolsets.translate import (
    _SILENT_PROGRESS,
    review_translated_file,
    translate_file,
)


async def test_translate_file_passes_silent_progress(monkeypatch, tmp_path):
    calls: list[dict] = []
    reports: list[tuple[str, str, bool]] = []
    path = tmp_path / "messages.po"
    path.write_text("", encoding="utf-8")

    def fake_translate_po(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("aitran.toolsets.translate.translate_po", fake_translate_po)

    ctx = SimpleNamespace(
        deps=OrchestratorDeps(tool_reporter=lambda *args: reports.append(args))
    )
    result = await translate_file(ctx, str(path), source_lang="en", target_lang="zh_CN")

    assert result == f"Translated PO file: {path}"
    assert calls[0]["progress"] is _SILENT_PROGRESS
    assert calls[0]["verbose"] is False
    assert reports == [("translate_file", f"Translated PO file: {path}", True)]


async def test_review_translated_file_passes_silent_progress(monkeypatch, tmp_path):
    calls: list[dict] = []
    reports: list[tuple[str, str, bool]] = []
    path = tmp_path / "messages.po"
    path.write_text("", encoding="utf-8")

    def fake_review_file(**kwargs):
        calls.append(kwargs)
        return {"pass": 1, "revise": 0, "reject": 0, "skip": 0}

    monkeypatch.setattr("aitran.toolsets.translate.review_file", fake_review_file)

    ctx = SimpleNamespace(
        deps=OrchestratorDeps(tool_reporter=lambda *args: reports.append(args))
    )
    result = await review_translated_file(
        ctx,
        str(path),
        source_lang="en",
        target_lang="zh_CN",
        auto_fix=True,
    )

    assert '"pass": 1' in result
    assert calls[0]["progress"] is _SILENT_PROGRESS
    assert calls[0]["auto_fix"] is True
    assert reports == [("review_translated_file", result, True)]
