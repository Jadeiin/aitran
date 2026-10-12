"""Tests for dictionary loading and matching."""

import json

import platformdirs
import pytest

from aitran.dicts import find_matching_entries, load_dictionary


@pytest.mark.parametrize("content", ['{"broken":', '["term"]', '{"term": 1}'])
def test_invalid_dictionary_reports_path_and_preserves_valid_fallback(
    content, monkeypatch, tmp_path, caplog
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        platformdirs, "user_config_dir", lambda *_args, **_kwargs: str(tmp_path)
    )
    config_dir = tmp_path / ".aitran"
    config_dir.mkdir()
    (config_dir / "dictionary.json").write_text(
        json.dumps({"term": "默认译文"}), encoding="utf-8"
    )
    invalid_path = config_dir / "dictionary-zh-cn.json"
    invalid_path.write_text(content, encoding="utf-8")

    assert load_dictionary("zh_CN") == {"term": "默认译文"}
    assert str(invalid_path) in caplog.text


def test_language_dictionary_overrides_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        platformdirs, "user_config_dir", lambda *_args, **_kwargs: str(tmp_path)
    )
    config_dir = tmp_path / ".aitran"
    config_dir.mkdir()
    (config_dir / "dictionary.json").write_text(
        json.dumps({"hello": "默认译文", "world": "世界"}), encoding="utf-8"
    )
    (config_dir / "dictionary-zh-cn.json").write_text(
        json.dumps({"hello": "你好"}), encoding="utf-8"
    )

    assert load_dictionary("zh_CN") == {"hello": "你好", "world": "世界"}


def test_load_empty_when_no_dictionary(monkeypatch, tmp_path):
    """Return empty when no dictionary files exist."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(
        platformdirs,
        "user_config_dir",
        lambda *_args, **_kwargs: str(empty),
    )

    result = load_dictionary("zz")
    assert result == {}


def test_find_matching_entries_basic(monkeypatch, tmp_path):
    """Find matching entries by case-insensitive substring."""
    dict_dir = tmp_path / "config"
    dict_dir.mkdir()
    dict_file = dict_dir / "dictionary-zz.json"
    dict_file.write_text(json.dumps({"hello": "你好", "world": "世界"}))

    monkeypatch.setattr(
        platformdirs,
        "user_config_dir",
        lambda *_args, **_kwargs: str(dict_dir),
    )

    sources = ["Hello world", "hello there"]
    result = find_matching_entries(sources, "zz")
    assert ("hello", "你好") in result
    assert ("world", "世界") in result


def test_find_matching_entries_no_match(monkeypatch, tmp_path):
    """Return empty when no keys match."""
    dict_dir = tmp_path / "config"
    dict_dir.mkdir()
    dict_file = dict_dir / "dictionary-zz.json"
    dict_file.write_text(json.dumps({"xyz": "abc"}))

    monkeypatch.setattr(
        platformdirs,
        "user_config_dir",
        lambda *_args, **_kwargs: str(dict_dir),
    )

    result = find_matching_entries(["hello world"], "zz")
    assert result == []
