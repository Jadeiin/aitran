"""Quality-assurance runner using translate-toolkit checkers."""

from __future__ import annotations

from dataclasses import dataclass, field

from aitran.qa.checkers import CATEGORY_LABELS, TeeChecker, build_checker


@dataclass(frozen=True, slots=True)
class QAError:
    """One rule-based QA failure on a translation unit."""

    checker: str
    """Checker name (e.g. ``"printf"``, ``"xmltags"``)."""

    message: str
    """Human-readable description of the failure."""

    severity: str
    """``"critical"``, ``"functional"``, ``"cosmetic"``, or ``"extraction"``."""


@dataclass(slots=True)
class UnitQAReport:
    """QA results for a single translation unit."""

    index: int
    """1-based unit index (matches the translation batch index)."""

    errors: list[QAError] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Whether this unit has any QA errors."""
        return len(self.errors) > 0


class QARunner:
    """Runs translate-toolkit quality checks on translation units.

    Usage::

        runner = QARunner(target_lang="zh_CN")
        reports = runner.check_units(units)
        for report in reports:
            if report.has_errors:
                for err in report.errors:
                    print(f"[{err.severity}] {err.checker}: {err.message}")
    """

    def __init__(
        self,
        *,
        target_lang: str | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        """Initialise the runner.

        Args:
            target_lang: Target language code (XPG/POSIX).
            exclude: Additional checker names to skip beyond the defaults.
        """
        self._checker: TeeChecker = build_checker(
            target_lang=target_lang, exclude=exclude
        )

    def check_unit(self, unit: object, index: int) -> UnitQAReport:
        """Run QA checks on a single translation unit.

        Args:
            unit: A translate-toolkit ``TranslationUnit`` (PO or XLIFF).
            index: 1-based index for this unit.

        Returns:
            A :class:`UnitQAReport` with zero or more :class:`QAError` entries.
        """
        failures = self._checker.run_filters(unit, categorised=True)
        errors: list[QAError] = []
        for check_name, info in failures.items():
            cat: int = info.get("category", 0)
            errors.append(
                QAError(
                    checker=check_name,
                    message=str(info.get("message", "")),
                    severity=CATEGORY_LABELS.get(cat, "other"),
                )
            )
        return UnitQAReport(index=index, errors=errors)

    def check_units(
        self, units: list[object], *, start_index: int = 1
    ) -> list[UnitQAReport]:
        """Run QA checks on a list of translation units.

        Args:
            units: Translate-toolkit ``TranslationUnit`` objects.
            start_index: 1-based starting index.

        Returns:
            One :class:`UnitQAReport` per unit, in the same order.
        """
        return [self.check_unit(u, start_index + i) for i, u in enumerate(units)]
