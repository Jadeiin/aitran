"""Dictionary management: load per-language dictionaries and match source strings."""

import json
import logging
import os

from aitran.utils import find_config, normalize_lang_code

logger = logging.getLogger(__name__)


def load_dictionary(lang: str) -> dict[str, str]:
    """Load the dictionary for a language.

    Resolve each filename through CWD/.aitran, git root/.aitran, then the
    platform-standard user config directory. Load dictionary.json first;
    dictionary-{normalized-lang}.json overrides entries with the same key.

    Returns:
        Merged dictionary. Empty dict if no dictionary found.
    """
    lang_code = normalize_lang_code(lang)
    lang_dict_path = find_config(f"dictionary-{lang_code}.json")
    default_dict_path = find_config("dictionary.json")

    result: dict[str, str] = {}
    paths = [default_dict_path]
    if lang_code:
        paths.append(lang_dict_path)
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
        except (ValueError, OSError) as error:
            logger.warning("Could not load dictionary %s: %s", path, error)
            continue
        if not isinstance(entries, dict) or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in entries.items()
        ):
            logger.warning(
                "Invalid dictionary %s: expected a JSON object mapping "
                "non-empty source strings to target strings",
                path,
            )
            continue
        result.update(entries)

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
