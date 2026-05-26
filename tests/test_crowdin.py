"""Tests for Crowdin helpers."""

from __future__ import annotations

import pytest
import requests
from crowdin_api.api_resources.enums import ExportProjectTranslationFormat

from aitran import crowdin


class _FakeTranslations:
    def __init__(
        self,
        export_payload: dict | None = None,
    ) -> None:
        self.last_export: dict | None = None
        self.export_payload = export_payload or {
            "data": {"url": "https://example.com/file.xliff"}
        }

    def export_project_translation(
        self,
        targetLanguageId,
        projectId=None,
        format=None,
        **kwargs,
    ):
        file_ids = kwargs.get("fileIds")
        self.last_export = {
            "targetLanguageId": targetLanguageId,
            "projectId": projectId,
            "format": format,
            "fileIds": file_ids,
        }
        return self.export_payload


class _FakeStorages:
    def add_storage(self, _handle):
        del _handle
        return {"data": {"id": 42}}


class _FakeProjects:
    def __init__(self) -> None:
        self.projects = [{"data": {"id": 1, "name": "demo"}}]

    def list_projects(self):
        return {"data": self.projects}


class _FakeSourceFiles:
    def __init__(self) -> None:
        self.files = [{"data": {"id": 2, "path": "/messages.xliff"}}]
        self.last_project_id: int | None = None

    def list_files(self, projectId=None):
        self.last_project_id = projectId
        return {"data": self.files}


class _FakeCrowdinClient:
    def __init__(self, *_args, export_payload: dict | None = None, **_kwargs) -> None:
        del _args
        self.kwargs = _kwargs
        self.projects = _FakeProjects()
        self.source_files = _FakeSourceFiles()
        self.translations = _FakeTranslations(export_payload=export_payload)
        self.storages = _FakeStorages()


class _FakeResponse:
    def __init__(self, content: bytes = b"data") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.crowdin.com/api/v2", "api.crowdin.com/api/v2/"),
        ("https://api.crowdin.com/api/v2/", "api.crowdin.com/api/v2/"),
        ("api.crowdin.com/api/v2", "api.crowdin.com/api/v2/"),
    ],
)
def test_normalize_crowdin_base_url(base_url, expected):
    assert crowdin._crowdin_base_url_parts(base_url)[0] == expected


def test_crowdin_download_writes_file(tmp_path, monkeypatch):
    output_path = tmp_path / "out.xliff"
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
        project=None,
        file_id=2,
        language="zh",
        output_path=str(output_path),
        organization=None,
        base_url=None,
        timeout_seconds=5,
    )

    assert output_path.read_bytes() == b"data"
    assert fake_client.translations.last_export == {
        "targetLanguageId": "zh",
        "projectId": 1,
        "format": ExportProjectTranslationFormat.XLIFF,
        "fileIds": [2],
    }


def test_crowdin_download_normalizes_base_url(tmp_path, monkeypatch):
    output_path = tmp_path / "out.xliff"
    fake_client = _FakeCrowdinClient()

    def _fake_get(*_args, **_kwargs):
        return _FakeResponse()

    def _factory(*_args, **kwargs):
        fake_client.kwargs = kwargs
        return fake_client

    monkeypatch.setattr(crowdin, "CrowdinClient", _factory)
    monkeypatch.setattr(crowdin.requests, "get", _fake_get)

    crowdin.download_translation(
        token="token",
        project_id=1,
        project=None,
        file_id=2,
        language="zh",
        output_path=str(output_path),
        organization=None,
        base_url="https://api.crowdin.com/api/v2",
        timeout_seconds=5,
    )

    assert fake_client.kwargs["base_url"] == "api.crowdin.com/api/v2/"
    assert fake_client.kwargs["http_protocol"] == "https"


def test_crowdin_download_request_error(monkeypatch, tmp_path):
    def _raise(*_args, **_kwargs):
        raise requests.RequestException("boom")

    fake_client = _FakeCrowdinClient()

    def _factory(*_args, **_kwargs):
        return fake_client

    monkeypatch.setattr(crowdin, "CrowdinClient", _factory)
    monkeypatch.setattr(crowdin.requests, "get", _raise)

    with pytest.raises(requests.RequestException, match="Failed to download"):
        crowdin.download_translation(
            token="token",
            project_id=1,
            project=None,
            file_id=2,
            language="zh",
            output_path=str(tmp_path / "out.xliff"),
            organization=None,
            base_url=None,
            timeout_seconds=5,
        )


def test_crowdin_download_resolves_project_name(tmp_path, monkeypatch):
    output_path = tmp_path / "out.xliff"
    fake_client = _FakeCrowdinClient()

    def _fake_get(*_args, **_kwargs):
        return _FakeResponse()

    def _factory(*_args, **_kwargs):
        return fake_client

    monkeypatch.setattr(crowdin, "CrowdinClient", _factory)
    monkeypatch.setattr(crowdin.requests, "get", _fake_get)

    crowdin.download_translation(
        token="token",
        project_id=None,
        project="demo",
        file_id=2,
        language="zh",
        output_path=str(output_path),
        organization=None,
        base_url=None,
        timeout_seconds=5,
    )

    assert fake_client.translations.last_export["projectId"] == 1


def test_crowdin_download_lists_files_when_file_id_missing(tmp_path, monkeypatch):
    fake_client = _FakeCrowdinClient()

    def _factory(*_args, **_kwargs):
        return fake_client

    monkeypatch.setattr(crowdin, "CrowdinClient", _factory)

    with pytest.raises(ValueError, match=r"2: /messages\.xliff"):
        crowdin.download_translation(
            token="token",
            project_id=None,
            project="demo",
            file_id=None,
            language="zh",
            output_path=str(tmp_path / "out.xliff"),
            organization=None,
            base_url=None,
            timeout_seconds=5,
        )
