"""Tests for orchestrator Crowdin toolset wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from aitran.toolsets._base import OrchestratorDeps
from aitran.toolsets.crowdin import (
    download_translation,
    upload_translation,
)


async def test_download_translation_reports_result_to_terminal(monkeypatch):
    calls: list[dict] = []
    reports: list[tuple[str, str, bool]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "aitran.toolsets.crowdin.crowdin_download",
        fake_download,
    )

    ctx = SimpleNamespace(
        deps=OrchestratorDeps(
            crowdin_token="token",
            tool_reporter=lambda *args: reports.append(args),
        )
    )

    result = await download_translation(
        ctx,
        project="demo",
        file_id=2,
        language="zh-CN",
        output_path="messages.xliff",
    )

    assert result == "Downloaded to messages.xliff"
    assert calls == [
        {
            "token": "token",
            "organization": None,
            "base_url": None,
            "timeout_seconds": 120,
            "file_id": 2,
            "language": "zh-CN",
            "output_path": "messages.xliff",
            "project_id": None,
            "project": "demo",
        }
    ]
    assert reports == [("crowdin_download_translation", result, True)]


async def test_upload_translation_reports_result_to_terminal(monkeypatch):
    calls: list[dict] = []
    reports: list[tuple[str, str, bool]] = []

    def fake_upload(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "aitran.toolsets.crowdin.crowdin_upload",
        fake_upload,
    )

    ctx = SimpleNamespace(
        deps=OrchestratorDeps(
            crowdin_token="token",
            tool_reporter=lambda *args: reports.append(args),
        )
    )

    result = await upload_translation(
        ctx,
        project="3",
        file_id=2,
        language="zh-CN",
        file_path="messages.xlf",
    )

    assert result == "Uploaded messages.xlf to Crowdin"
    assert calls == [
        {
            "token": "token",
            "organization": None,
            "base_url": None,
            "timeout_seconds": 120,
            "file_id": 2,
            "language": "zh-CN",
            "file_path": "messages.xlf",
            "project_id": 3,
            "project": None,
        }
    ]
    assert reports == [("crowdin_upload_translation", result, True)]
