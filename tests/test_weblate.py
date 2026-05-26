"""Tests for Weblate helpers."""

from __future__ import annotations

from aitran import weblate


class _FakeWeblate:
    def __init__(self, key: str, url: str) -> None:
        self.key = key
        self.url = url
        self.last_method: str | None = None
        self.last_path: str | None = None
        self.last_params: dict | None = None
        self.last_data: dict | None = None

    def raw_request(self, method: str, path: str, params: dict | None = None) -> bytes:
        self.last_method = method
        self.last_path = path
        self.last_params = params
        return b"payload"

    def request(self, method: str, path: str, **kwargs):
        self.last_method = method
        self.last_path = path
        self.last_data = kwargs.get("data")
        return {"ok": True}


def test_weblate_download_writes_file(tmp_path, monkeypatch):
    output_path = tmp_path / "nested" / "messages.po"
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    weblate.download_translation(
        url="https://example.com",
        token="token",
        project="project",
        component="component",
        language="zh",
        output_path=str(output_path),
        convert=None,
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.url == "https://example.com/api/"
    assert fake.last_method == "GET"
    assert fake.last_path == "translations/project/component/zh/file/"
    assert fake.last_params is None


def test_weblate_download_convert(tmp_path, monkeypatch):
    output_path = tmp_path / "messages.xlf"
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    weblate.download_translation(
        url="https://example.com",
        token="token",
        project="project",
        component="component",
        language="zh",
        output_path=str(output_path),
        convert="xliff",
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.last_params == {"format": "xliff"}


def test_weblate_upload_sets_method_and_fuzzy(tmp_path, monkeypatch):
    upload_path = tmp_path / "messages.po"
    upload_path.write_bytes(b"content")
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    weblate.upload_translation(
        url="https://example.com",
        token="token",
        project="project",
        component="component",
        language="zh",
        file_path=str(upload_path),
        method="replace",
        fuzzy="process",
    )

    assert fake.last_method == "POST"
    assert fake.last_path == "translations/project/component/zh/upload/"
    assert fake.last_data == {"method": "replace", "fuzzy": "process"}
