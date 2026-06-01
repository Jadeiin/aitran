"""Checker configuration for translate-toolkit quality checks."""

from __future__ import annotations

import os

import autocorrect_py as ac
from translate.filters.checks import (
    CheckerConfig,
    FilterFailure,
    StandardChecker,
    StandardUnitChecker,
    TeeChecker,
)
from translate.filters.decorators import cosmetic, functional

from aitran.utils import _git_root

_autocorrect_config_loaded = False


def _load_autocorrect_config() -> None:
    """Load .autocorrectrc from CWD or git root (once)."""
    global _autocorrect_config_loaded
    if _autocorrect_config_loaded:
        return
    _autocorrect_config_loaded = True

    for directory in (os.getcwd(), _git_root()):
        if directory is None:
            continue
        path = os.path.join(directory, ".autocorrectrc")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                ac.load_config(f.read())
            return


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


def _run_autocorrect(text: str, *, severity: ac.Severity) -> bool:
    """Run autocorrect lint and raise FilterFailure for matching severity.

    Args:
        text: Target text to lint.
        severity: autocorrect severity to match.

    Returns:
        True if no findings at this severity.

    Raises:
        FilterFailure: If autocorrect finds issues at *severity*.
    """
    _load_autocorrect_config()

    result = ac.lint_for(text, "txt")
    messages = [
        f'"{lr.old}" → "{lr.new}"' for lr in result.lines if lr.severity == severity
    ]
    if messages:
        raise FilterFailure(messages)
    return True


class _AITranChecker(StandardChecker):
    """Extended checker that treats full-width brackets as equivalents."""

    @functional
    def autocorrect_error(self, str1: str, str2: str) -> bool:  # noqa: ARG002
        """Check CJK half-width/full-width spacing (error-level).

        Returns:
            True if no error-level findings.
        """
        return _run_autocorrect(str2, severity=ac.Severity.Error)

    @cosmetic
    def autocorrect_warning(self, str1: str, str2: str) -> bool:  # noqa: ARG002
        """Check CJK half-width/full-width spacing (warning-level).

        Returns:
            True if no warning-level findings.
        """
        return _run_autocorrect(str2, severity=ac.Severity.Warning)

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
