"""Checker configuration for translate-toolkit quality checks."""

from __future__ import annotations

from translate.filters.checks import (
    CheckerConfig,
    StandardChecker,
    StandardUnitChecker,
    TeeChecker,
)
from translate.filters.decorators import cosmetic

# Noisy or niche checkers excluded by default.  Everything else from
# StandardChecker + StandardUnitChecker is enabled — the LLM reviewer
# decides which failures are real.
DEFAULT_EXCLUDE: set[str] = {
    "spellcheck",  # dictionary-based, high false-positive rate
    "validchars",  # encoding issues, not translation quality
    "compendiumconflicts",  # merge-tool artefact, irrelevant for LLM review
    "credits",  # source-credits metadata, not a translation check
    "kdecomments",  # KDE-specific comment format
    "hassuggestion",  # internal tool state, not a quality signal
}


class _AITranChecker(StandardChecker):
    """Extended checker that treats full-width brackets as equivalents."""

    @cosmetic
    def brackets(self, str1, str2) -> bool:
        """Check that the number of brackets in both strings match.

        Counts half-width and full-width equivalents as the same bracket
        type so that ``（foo）`` in the source is satisfied by ``(foo)``
        in the target.

        Returns:
            True if bracket counts match.

        Raises:
            FilterFailure: If bracket counts differ.
        """
        str1 = self.filtervariables(str1)
        str2 = self.filtervariables(str2)

        messages = []
        missing = []
        extra = []

        # Each group: (tuple of opens, tuple of closes).
        # All characters in a group are considered equivalent.
        bracket_groups = [
            # parentheses
            (("(", "（"), (")", "）")),
            # square bracket family
            (("[", "【", "「", "『", "〔"), ("]", "】", "」", "』", "〕")),
            # curly braces
            (("{", "｛"), ("}", "｝")),
            # angle bracket family
            (("<", "〈", "《"), (">", "〉", "》")),
        ]

        for opens, closes in bracket_groups:
            open1 = sum(str1.count(ch) for ch in opens)
            close1 = sum(str1.count(ch) for ch in closes)
            open2 = sum(str2.count(ch) for ch in opens)
            close2 = sum(str2.count(ch) for ch in closes)

            if open2 < open1:
                missing.append(f"'{opens[0]}' open")
            elif open2 > open1:
                extra.append(f"'{opens[0]}' open")

            if close2 < close1:
                missing.append(f"'{closes[0]}' close")
            elif close2 > close1:
                extra.append(f"'{closes[0]}' close")

        if missing:
            messages.append(f"Missing {', '.join(missing)}")
        if extra:
            messages.append(f"Added {', '.join(extra)}")
        if messages:
            from translate.filters.checks import FilterFailure

            raise FilterFailure(messages)

        return True


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
        checkerclasses=[_AITranChecker, StandardUnitChecker],
        excludefilters=excluded,
        languagecode=target_lang,
    )
