"""Checker configuration for translate-toolkit quality checks."""

from __future__ import annotations

from translate.filters.checks import (
    CheckerConfig,
    StandardChecker,
    StandardUnitChecker,
    TeeChecker,
)

# Noisy or niche checkers excluded by default.  Everything else from
# StandardChecker + StandardUnitChecker is enabled — the LLM reviewer
# decides which failures are real.
DEFAULT_EXCLUDE: set[str] = {
    "spellcheck",           # dictionary-based, high false-positive rate
    "validchars",           # encoding issues, not translation quality
    "compendiumconflicts",  # merge-tool artefact, irrelevant for LLM review
    "credits",              # source-credits metadata, not a translation check
    "kdecomments",          # KDE-specific comment format
    "hassuggestion",        # internal tool state, not a quality signal
}

# Severity labels for the four translate-toolkit categories.
CATEGORY_LABELS: dict[int, str] = {
    100: "critical",
    60: "functional",
    30: "cosmetic",
    10: "extraction",
    0: "other",
}


def build_checker(
    *,
    target_lang: str | None = None,
    exclude: set[str] | None = None,
) -> TeeChecker:
    """Build a TeeChecker with broad coverage.

    Args:
        target_lang: Target language code (XPG/POSIX).  Used for
            language-specific ignore rules inside translate-toolkit.
        exclude: Additional checker names to skip.  Merged with
            :data:`DEFAULT_EXCLUDE`.

    Returns:
        A configured :class:`~translate.filters.checks.TeeChecker`.
    """
    excluded = DEFAULT_EXCLUDE | (exclude or set())

    config = CheckerConfig()
    if target_lang:
        config.targetlanguage = target_lang

    return TeeChecker(
        checkerconfig=config,
        checkerclasses=[StandardChecker, StandardUnitChecker],
        excludefilters=excluded,
        languagecode=target_lang,
    )
