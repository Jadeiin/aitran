"""Tests for Crowdin helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests
from crowdin_api.api_resources.enums import ExportProjectTranslationFormat

from aitran import crowdin


class _FakeRequester:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self._statuses = iter(statuses or ["finished"])

    def request(self, method, path, **_kwargs):
        del path
        assert method == "get"
        try:
            status = next(self._statuses)
        except StopIteration:
            status = "inProgress"
        payload = {"data": {"status": status}}
        if status == "finished":
            payload["data"]["url"] = "https://example.com/file.po"
        return payload


class _FakeTranslations:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.requester = _FakeRequester(statuses)
        self.last_export: dict | None = None

    def export_project_translation(
        self,
        targetLanguageId,
        projectId=None,
        format=None,
        fileIds=None,
        **_kwargs,
    ):
        self.last_export = {
            "targetLanguageId": targetLanguageId,
            "projectId": projectId,
            "format": format,
            "fileIds": fileIds,
        }
        return {"data": {"identifier": "export-17"}}


class _FakeStorages:
    def add_storage(self, _handle):
        del _handle
        return {"data": {"id": 42}}


class _FakeCrowdinClient:
    def __init__(self, *_args, **_kwargs) -> None:
        del _args, _kwargs
        self.translations = _FakeTranslations()
        self.storages = _FakeStorages()


class _FakeResponse:
    def __init__(self, content: bytes = b"data") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_wait_for_export_finished():
    client = SimpleNamespace(translations=_FakeTranslations(["finished"]))
    url = crowdin._wait_for_export(
        client,
        export_id="export",
        project_id=2,
        timeout_seconds=5,
        poll_interval=1,
    )
    assert url == "https://example.com/file.po"


def test_wait_for_export_failed():
    client = SimpleNamespace(translations=_FakeTranslations(["failed"]))
    with pytest.raises(ValueError, match="ended with status"):
        crowdin._wait_for_export(
            client,
            export_id="export",
            project_id=2,
            timeout_seconds=5,
            poll_interval=1,
        )


def test_wait_for_export_canceled():
    client = SimpleNamespace(translations=_FakeTranslations(["canceled"]))
    with pytest.raises(ValueError, match="ended with status"):
        crowdin._wait_for_export(
            client,
            export_id="export",
            project_id=2,
            timeout_seconds=5,
            poll_interval=1,
        )


def test_wait_for_export_timeout(monkeypatch):
    class _FakeTime:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    fake_time = _FakeTime()
    monkeypatch.setattr(crowdin.time, "monotonic", fake_time.monotonic)
    monkeypatch.setattr(crowdin.time, "sleep", fake_time.sleep)

    client = SimpleNamespace(translations=_FakeTranslations(["inProgress"]))
    with pytest.raises(TimeoutError, match="Timed out"):
        crowdin._wait_for_export(
            client,
            export_id="export",
            project_id=2,
            timeout_seconds=2,
            poll_interval=1,
        )


def test_crowdin_download_writes_file(tmp_path, monkeypatch):
    output_path = tmp_path / "out.po"
    fake_client = _FakeCrowdinClient()

    def _fake_get(*_args, **_kwargs):
        return _FakeResponse()

    def _factory(*_args, **_kwargs):
        return fake_client

    monkeypatch.setattr(crowdin, "CrowdinClient", _factory)
    monkeypatch.setattr(crowdin.requests, "get", _fake_get)

    crowdin.download_translation(
        token="token",
        project_id=1,
        file_id=2,
        language="zh",
        export_format=ExportProjectTranslationFormat.XLIFF,
        output_path=str(output_path),
        organization=None,
        base_url=None,
        timeout_seconds=5,
        poll_interval=1,
    )

    assert output_path.read_bytes() == b"data"
    assert fake_client.translations.last_export == {
        "targetLanguageId": "zh",
        "projectId": 1,
        "format": ExportProjectTranslationFormat.XLIFF,
        "fileIds": [2],
    }


def test_crowdin_download_request_error(monkeypatch, tmp_path):
    def _raise(*_args, **_kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(crowdin, "CrowdinClient", _FakeCrowdinClient)
    monkeypatch.setattr(crowdin.requests, "get", _raise)

    with pytest.raises(requests.RequestException, match="Failed to download"):
        crowdin.download_translation(
            token="token",
            project_id=1,
            file_id=2,
            language="zh",
            export_format=ExportProjectTranslationFormat.XLIFF,
            output_path=str(tmp_path / "out.po"),
            organization=None,
            base_url=None,
            timeout_seconds=5,
            poll_interval=1,
        )
