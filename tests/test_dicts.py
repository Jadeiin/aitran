"""Tests for dictionary loading and matching."""

import json

import platformdirs

from aitran.dicts import find_matching_entries, load_dictionary


def test_load_empty_when_no_dictionary(monkeypatch, tmp_path):
    """Return empty when no dictionary files exist."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(
        platformdirs,
        "user_config_dir",
        lambda appname, ensure_exists=False: str(empty),
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
        lambda appname, ensure_exists=False: str(dict_dir),
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
        lambda appname, ensure_exists=False: str(dict_dir),
    )

    result = find_matching_entries(["hello world"], "zz")
    assert result == []
