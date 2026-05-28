"""Tests for review apply_batch on PO and XLIFF adapters."""

from translate.storage import po

from aitran.agents.reviewer import ReviewedUnit
from aitran.translate import PoTranslator


def _po(content: str) -> po.pofile:
    return po.pofile.parsestring(content.encode())


def _indexed(units):
    return dict(enumerate(units, start=1))


class TestPoReviewApply:
    def test_revise_marks_fuzzy_with_note(self):
        pofile = _po('#: src/a.py:1\nmsgid "Hello %s"\nmsgstr "你好"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="revise",
                corrected="你好 %s",
                note="missing placeholder %s",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results)
        assert unit.isfuzzy()
        assert unit.target == "你好"  # not changed without auto_fix

    def test_revise_auto_fix_writes_correction(self):
        pofile = _po('#: src/a.py:1\nmsgid "Hello %s"\nmsgstr "你好"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="revise",
                corrected="你好 %s",
                note="missing placeholder %s",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results, auto_fix=True)
        assert not unit.isfuzzy()
        assert unit.target == "你好 %s"

    def test_auto_fix_decodes_xml_escaped_correction(self):
        pofile = _po('#: src/a.py:1\nmsgid "Click <b>here</b>"\nmsgstr "点击"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="revise",
                corrected="点击&lt;b&gt;此处&lt;/b&gt;",
                note="missing markup",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results, auto_fix=True)
        assert unit.target == "点击<b>此处</b>"

    def test_reject_without_correction_marks_fuzzy(self):
        pofile = _po('#: src/a.py:1\nmsgid "Hello"\nmsgstr "你好"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="reject",
                note="meaning is wrong",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results)
        assert unit.isfuzzy()

    def test_reject_auto_fix_without_correction_keeps_fuzzy(self):
        pofile = _po('#: src/a.py:1\nmsgid "Hello"\nmsgstr "你好"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="reject",
                note="meaning is wrong",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results, auto_fix=True)
        # reject without corrected → still fuzzy, target unchanged
        assert unit.isfuzzy()

    def test_reject_auto_fix_with_correction_writes(self):
        pofile = _po('#: src/a.py:1\nmsgid "Hello %s"\nmsgstr "你好"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="reject",
                corrected="你好 %s",
                note="placeholder missing",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results, auto_fix=True)
        assert not unit.isfuzzy()
        assert unit.target == "你好 %s"

    def test_review_note_is_added(self):
        pofile = _po('#: src/a.py:1\nmsgid "Hello"\nmsgstr "你好"\n')
        unit = pofile.units[0]
        results = [
            ReviewedUnit(
                index=1,
                verdict="revise",
                note="check punctuation",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results)
        notes = unit.getnotes()
        assert "review" in notes.lower()
        assert "check punctuation" in notes

    def test_mixed_results(self):
        pofile = _po(
            '#: src/a.py:1\nmsgid "Hello"\nmsgstr "你好"\n\n'
            '#: src/a.py:2\nmsgid "World"\nmsgstr "世界"\n\n'
            '#: src/a.py:3\nmsgid "Error"\nmsgstr "错误"\n'
        )
        units = [u for u in pofile.units if u.source]
        # Only revise/reject results; unit 1 is implicitly OK (not in results)
        results = [
            ReviewedUnit(index=2, verdict="revise", corrected="修正", note="fix"),
            ReviewedUnit(index=3, verdict="reject", note="wrong"),
        ]
        PoTranslator.apply_review_batch(
            pofile,
            {2: units[1], 3: units[2]},
            results,
        )
        assert not units[0].isfuzzy()
        assert units[1].isfuzzy()
        assert units[2].isfuzzy()

    def test_sparse_results_skip_clean_units(self):
        pofile = _po(
            '#: src/a.py:1\nmsgid "Hello"\nmsgstr "你好"\n\n'
            '#: src/a.py:2\nmsgid "World"\nmsgstr "世界"\n\n'
            '#: src/a.py:3\nmsgid "Error"\nmsgstr "错误"\n'
        )
        units = [u for u in pofile.units if u.source]
        # Only unit 3 has a problem; units 1 and 2 are clean (omitted)
        results = [ReviewedUnit(index=3, verdict="reject", note="wrong")]
        PoTranslator.apply_review_batch(pofile, _indexed(units), results)
        assert not units[0].isfuzzy()
        assert not units[1].isfuzzy()
        assert units[2].isfuzzy()

    def test_auto_fix_plural_preserves_other_forms(self):
        pofile = _po(
            "#: src/a.py:1\n"
            'msgid "apple"\n'
            'msgid_plural "apples"\n'
            'msgstr[0] "苹果"\n'
            'msgstr[1] "苹果们"\n'
        )
        unit = pofile.units[0]
        assert unit.hasplural()
        results = [
            ReviewedUnit(
                index=1,
                verdict="revise",
                corrected="修正苹果",
                note="fix first form",
            )
        ]
        PoTranslator.apply_review_batch(pofile, {1: unit}, results, auto_fix=True)
        assert not unit.isfuzzy()
        targets = unit.target.strings
        assert targets[0] == "修正苹果"
        assert targets[1] == "苹果们"
