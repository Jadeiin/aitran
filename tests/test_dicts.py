"""Tests for dictionary loading and matching."""

import json
import os
import tempfile

from aitran.dicts import find_matching_entries, load_dictionary


def test_load_empty_when_no_dictionary(monkeypatch):
    """Return empty dict when no dictionary files exist."""
    monkeypatch.setenv("HOME", tempfile.mkdtemp())
    result = load_dictionary("zz")
    assert result == {}


def test_find_matching_entries_basic(monkeypatch, tmp_path):
    """Find matching entries by case-insensitive substring."""
    dict_dir = tmp_path / ".aitran"
    dict_dir.mkdir()
    dict_file = dict_dir / "dictionary-zz.json"
    dict_file.write_text(json.dumps({"hello": "你好", "world": "世界"}))

    monkeypatch.setenv("HOME", str(tmp_path))

    sources = ["Hello world", "hello there"]
    result = find_matching_entries(sources, "zz")
    assert ("hello", "你好") in result
    assert ("world", "世界") in result


def test_find_matching_entries_no_match(monkeypatch, tmp_path):
    """Return empty when no keys match."""
    dict_dir = tmp_path / ".aitran"
    dict_dir.mkdir()
    dict_file = dict_dir / "dictionary-zz.json"
    dict_file.write_text(json.dumps({"xyz": "abc"}))

    monkeypatch.setenv("HOME", str(tmp_path))

    result = find_matching_entries(["hello world"], "zz")
    assert result == []
