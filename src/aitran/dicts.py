"""Dictionary management: load per-language dictionaries and match source strings."""

import json
import os

from aitran.utils import find_config, normalize_lang_code


def load_dictionary(lang: str) -> dict[str, str]:
    """Load the dictionary for a language.

    Cascading lookup:
    1. ~/.aitran/dictionary-{lang}.json
    2. ~/.aitran/dictionary.json (default)

    Returns:
        Merged dictionary. Empty dict if no dictionary found.
    """
    lang_code = normalize_lang_code(lang)
    lang_dict_path = find_config(f"dictionary-{lang_code}.json")
    default_dict_path = find_config("dictionary.json")

    result: dict[str, str] = {}
    if os.path.exists(default_dict_path):
        try:
            with open(default_dict_path, encoding="utf-8") as f:
                result.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    if lang_code and os.path.exists(lang_dict_path):
        try:
            with open(lang_dict_path, encoding="utf-8") as f:
                result.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    return result


def find_matching_entries(sources: list[str], lang: str) -> list[tuple[str, str]]:
    """Find dictionary entries whose key appears as a case-insensitive substring.

    Returns:
        List of (key, value) pairs.
    """
    dictionary = load_dictionary(lang)
    if not dictionary:
        return []

    matched: list[tuple[str, str]] = []
    for key, value in dictionary.items():
        key_lower = key.lower()
        if any(key_lower in src.lower() for src in sources):
            matched.append((key, value))

    return matched
