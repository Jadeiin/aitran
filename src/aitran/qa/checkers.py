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

        # (half-width open/close, full-width open/close)
        bracket_pairs = [
            ("(", ")", "（", "）"),  # parentheses
            ("[", "]", "【", "】"),  # square brackets
            ("{", "}", "｛", "｝"),  # curly braces
            ("<", ">", "〈", "〉"),  # angle brackets
            ("<", ">", "《", "》"),  # double angle
            ("[", "]", "「", "」"),  # CJK corner brackets
            ("[", "]", "『", "』"),  # CJK white corner
            ("[", "]", "〔", "〕"),  # tortoise shell
        ]

        seen: set[tuple[str, str]] = set()
        for half_open, half_close, fw_open, fw_close in bracket_pairs:
            key = (half_open, half_close)
            if key in seen:
                continue
            seen.add(key)

            count1 = (
                str1.count(half_open)
                + str1.count(half_close)
                + str1.count(fw_open)
                + str1.count(fw_close)
            )
            count2 = (
                str2.count(half_open)
                + str2.count(half_close)
                + str2.count(fw_open)
                + str2.count(fw_close)
            )

            if count2 < count1:
                missing.append(f"'{half_open}' or equivalent")
            elif count2 > count1:
                extra.append(f"'{half_open}' or equivalent")

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
