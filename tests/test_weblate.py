"""Tests for Weblate helpers."""

from __future__ import annotations

import pytest

from aitran import weblate


class _FakeWeblate:
    def __init__(self, key: str, url: str) -> None:
        self.key = key
        self.url = url
        self.last_object_path: str | None = None
        self.last_raw_request: tuple[str, str] | None = None
        self.translation = _FakeTranslation()
        self.translation.weblate = self

    def get_object(self, path: str):
        self.last_object_path = path
        return self.translation

    def list_projects(self):
        return [{"slug": "project", "name": "Project"}]

    def raw_request(self, method: str, url: str) -> bytes:
        self.last_raw_request = (method, url)
        return b"payload"


class _FakeTranslation:
    def __init__(self) -> None:
        self.last_data: dict | None = None
        self.last_download: str | None = None
        self.list_result = [{"language_code": "zh", "translated_percent": 50}]
        self.stats_result = {"total": 10, "translated": 5}
        self.weblate: _FakeWeblate | None = None

    def _get_stored(self, key: str) -> str:
        assert key == "file_url"
        return "https://example.com/file/"

    def download(self, convert: str | None = None) -> bytes:
        self.last_download = convert
        return b"payload"

    def list(self):
        return self.list_result

    def statistics(self):
        return self.stats_result

    def __iter__(self):
        return iter(())

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
        download_format=None,
        untranslated_only=False,
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.url == "https://example.com/api/"
    assert fake.last_object_path == "project/component/zh"
    assert fake.translation.last_download == "po"


def test_weblate_list_objects(monkeypatch):
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    assert weblate.list_objects(
        url="https://example.com",
        token="token",
        object_path="project/component",
    ) == [{"language_code": "zh", "translated_percent": 50}]


def test_weblate_list_objects_wraps_translation_object(monkeypatch):
    fake = _FakeWeblate(key="token", url="https://example.com/api/")
    fake.translation.list_result = fake.translation

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    assert weblate.list_objects(
        url="https://example.com",
        token="token",
        object_path="project/component/zh",
    ) == [fake.translation]


def test_weblate_get_stats(monkeypatch):
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    assert weblate.get_stats(
        url="https://example.com",
        token="token",
        object_path="project/component/zh",
    ) == {"total": 10, "translated": 5}


def test_weblate_get_stats_materializes_component_iterator(monkeypatch):
    fake = _FakeWeblate(key="token", url="https://example.com/api/")
    fake.translation.stats_result = iter([
        {"language_code": "zh", "translated": 5},
        {"language_code": "fr", "translated": 3},
    ])

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)

    assert weblate.get_stats(
        url="https://example.com",
        token="token",
        object_path="project/component",
    ) == [
        {"language_code": "zh", "translated": 5},
        {"language_code": "fr", "translated": 3},
    ]


def test_weblate_download_format(tmp_path, monkeypatch):
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
        download_format="xliff11",
        untranslated_only=True,
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.last_raw_request == (
        "get",
        "https://example.com/file/?format=xliff11&q=is%3Auntranslated",
    )


def test_weblate_untranslated_download_uses_compatibility_layer(monkeypatch):
    fake = _FakeTranslation()
    fake.weblate = _FakeWeblate(key="token", url="https://example.com/api/")
    captured = {}

    def _file_url(translation):
        captured["translation"] = translation
        return "https://example.com/file/"

    monkeypatch.setattr(weblate, "_translation_file_url", _file_url)

    assert (
        weblate._download_translation_content(
            fake,
            "po",
            untranslated_only=True,
        )
        == b"payload"
    )
    assert captured["translation"] is fake
    assert fake.weblate.last_raw_request == (
        "get",
        "https://example.com/file/?format=po&q=is%3Auntranslated",
    )


def test_weblate_download_rejects_invalid_explicit_output_extension(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "messages.txt"
    fake = _FakeWeblate(key="token", url="https://example.com/api/")

    def _factory(*, key, url):
        fake.key = key
        fake.url = url
        return fake

    monkeypatch.setattr(weblate, "Weblate", _factory)
    monkeypatch.setattr(weblate, "Translation", _FakeTranslation)

    with pytest.raises(
        ValueError,
        match=r"Only \.po, \.xliff, or \.xlf files are supported\.",
    ):
        weblate.download_translation(
            url="https://example.com",
            token="token",
            object_path="project/component/zh",
            output_path=str(output_path),
            download_format="po",
            untranslated_only=False,
        )


def test_weblate_download_xlf_uses_xliff_format(tmp_path, monkeypatch):
    output_path = tmp_path / "messages.xlf"
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
        download_format=None,
        untranslated_only=False,
    )

    assert fake.translation.last_download == "xliff"


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
