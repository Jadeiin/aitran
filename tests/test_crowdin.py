"""Tests for Crowdin helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from aitran import crowdin


class _FakeTranslations:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self._statuses = iter(statuses or ["finished"])

    def build_project_file_translation(self, file_id, language, projectId=None):
        del file_id, language, projectId
        return {"data": {"id": 17}}

    def check_project_build_status(self, build_id, projectId=None):
        del build_id, projectId
        try:
            status = next(self._statuses)
        except StopIteration:
            status = "inProgress"
        return {"data": {"status": status}}

    def download_project_translations(self, build_id, projectId=None):
        del build_id, projectId
        return {"data": {"url": "https://example.com/file.po"}}

    def upload_translation(self, language, storage_id, file_id, projectId=None):
        del language, storage_id, file_id, projectId
        return {"data": {"id": 1}}


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


def test_wait_for_build_finished():
    client = SimpleNamespace(translations=_FakeTranslations(["finished"]))
    crowdin._wait_for_build(
        client,
        build_id=1,
        project_id=2,
        timeout_seconds=5,
        poll_interval=1,
    )


def test_wait_for_build_failed():
    client = SimpleNamespace(translations=_FakeTranslations(["failed"]))
    with pytest.raises(ValueError, match="ended with status"):
        crowdin._wait_for_build(
            client,
            build_id=1,
            project_id=2,
            timeout_seconds=5,
            poll_interval=1,
        )


def test_wait_for_build_canceled():
    client = SimpleNamespace(translations=_FakeTranslations(["canceled"]))
    with pytest.raises(ValueError, match="ended with status"):
        crowdin._wait_for_build(
            client,
            build_id=1,
            project_id=2,
            timeout_seconds=5,
            poll_interval=1,
        )


def test_wait_for_build_timeout(monkeypatch):
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
        crowdin._wait_for_build(
            client,
            build_id=1,
            project_id=2,
            timeout_seconds=2,
            poll_interval=1,
        )


def test_crowdin_download_writes_file(tmp_path, monkeypatch):
    output_path = tmp_path / "out.po"

    def _fake_get(*_args, **_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(crowdin, "CrowdinClient", _FakeCrowdinClient)
    monkeypatch.setattr(crowdin.requests, "get", _fake_get)

    crowdin.download_translation(
        token="token",
        project_id=1,
        file_id=2,
        language="zh",
        output_path=str(output_path),
        organization=None,
        base_url=None,
        timeout_seconds=5,
        poll_interval=1,
    )

    assert output_path.read_bytes() == b"data"


def test_crowdin_download_request_error(monkeypatch, tmp_path):
    def _raise(*_args, **_kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(crowdin, "CrowdinClient", _FakeCrowdinClient)
    monkeypatch.setattr(crowdin.requests, "get", _raise)

    with pytest.raises(requests.RequestException, match="boom"):
        crowdin.download_translation(
            token="token",
            project_id=1,
            file_id=2,
            language="zh",
            output_path=str(tmp_path / "out.po"),
            organization=None,
            base_url=None,
            timeout_seconds=5,
            poll_interval=1,
        )
