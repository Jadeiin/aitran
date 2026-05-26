"""Tests for Weblate helpers."""

from __future__ import annotations

from aitran import weblate


class _FakeWeblate:
    def __init__(self, key: str, url: str) -> None:
        self.key = key
        self.url = url
        self.last_object_path: str | None = None
        self.translation = _FakeTranslation()

    def get_object(self, path: str):
        self.last_object_path = path
        return self.translation


class _FakeTranslation:
    def __init__(self) -> None:
        self.last_convert: str | None = None
        self.last_data: dict | None = None

    def download(self, convert: str | None = None) -> bytes:
        self.last_convert = convert
        return b"payload"

    def upload(self, _file, **kwargs):
        self.last_data = kwargs
        return {"ok": True}


def test_weblate_download_writes_file(tmp_path, monkeypatch):
    output_path = tmp_path / "nested" / "messages.po"
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)
    monkeypatch.setattr(weblate, "Translation", _FakeTranslation)

    weblate.download_translation(
        url="https://example.com",
        token="token",
        object_path="project/component/zh",
        output_path=str(output_path),
        convert=None,
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.url == "https://example.com/api/"
    assert fake.last_object_path == "project/component/zh"
    assert fake.translation.last_convert is None


def test_weblate_download_convert(tmp_path, monkeypatch):
    output_path = tmp_path / "messages.xliff"
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)
    monkeypatch.setattr(weblate, "Translation", _FakeTranslation)

    weblate.download_translation(
        url="https://example.com",
        token="token",
        object_path="project/component/zh",
        output_path=str(output_path),
        convert="xliff",
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.translation.last_convert == "xliff"


def test_weblate_upload_sets_method_and_fuzzy(tmp_path, monkeypatch):
    upload_path = tmp_path / "messages.po"
    upload_path.write_bytes(b"content")
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)
    monkeypatch.setattr(weblate, "Translation", _FakeTranslation)

    weblate.upload_translation(
        url="https://example.com",
        token="token",
        object_path="project/component/zh",
        file_path=str(upload_path),
        method="replace",
        fuzzy="process",
    )

    assert fake.last_object_path == "project/component/zh"
    assert fake.translation.last_data == {"method": "replace", "fuzzy": "process"}
