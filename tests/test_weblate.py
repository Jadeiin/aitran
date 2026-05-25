"""Tests for Weblate helpers."""

from __future__ import annotations

from aitran import weblate


class _FakeWeblate:
    def __init__(self, key: str, url: str) -> None:
        self.key = key
        self.url = url
        self.last_method: str | None = None
        self.last_path: str | None = None

    def raw_request(self, method: str, path: str) -> bytes:
        self.last_method = method
        self.last_path = path
        return b"payload"


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
    )

    assert output_path.read_bytes() == b"payload"
    assert fake.url == "https://example.com/api/"
    assert fake.last_method == "GET"
    assert fake.last_path == "translations/project/component/zh/file/"
