"""Tests for the QA runner."""

from tests.helpers import po_parse as _po


class TestQARunner:
    def test_clean_unit_has_no_errors(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Hello"\nmsgstr "你好"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        assert not report.has_errors
        assert report.errors == []

    def test_missing_placeholder_detected(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Hello %s"\nmsgstr "你好"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        assert report.has_errors
        checker_names = {e.checker for e in report.errors}
        assert "printf" in checker_names

    def test_xml_tag_mismatch_detected(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Click <b>here</b>"\nmsgstr "点击"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        assert report.has_errors
        checker_names = {e.checker for e in report.errors}
        assert "xmltags" in checker_names

    def test_bracket_mismatch_detected(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Value (ok)"\nmsgstr "值"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        assert report.has_errors
        checker_names = {e.checker for e in report.errors}
        assert "brackets" in checker_names

    def test_fullwidth_bracket_accepted(self):
        from aitran.qa import QARunner

        po = _po(
            '#: src/app.py:1\nmsgid "Value (ok)"\nmsgstr "值（好）"\n'
        )
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        checker_names = {e.checker for e in report.errors}
        assert "brackets" not in checker_names

    def test_fullwidth_to_halfwidth_bracket_accepted(self):
        from aitran.qa import QARunner

        po = _po(
            '#: src/app.py:1\nmsgid "值（好）"\nmsgstr "Value (ok)"\n'
        )
        unit = po.units[0]
        runner = QARunner(target_lang="en")
        report = runner.check_unit(unit, index=1)
        checker_names = {e.checker for e in report.errors}
        assert "brackets" not in checker_names

    def test_endpunc_mismatch_detected(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Error."\nmsgstr "错误"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        assert report.has_errors
        checker_names = {e.checker for e in report.errors}
        assert "endpunc" in checker_names

    def test_check_units_batch(self):
        from aitran.qa import QARunner

        po = _po(
            '#: src/app.py:1\nmsgid "Hello"\nmsgstr "你好"\n\n'
            '#: src/app.py:2\nmsgid "Hello %s"\nmsgstr "你好"\n'
        )
        units = [u for u in po.units if u.source]
        runner = QARunner(target_lang="zh_CN")
        reports = runner.check_units(units, start_index=1)
        assert len(reports) == 2
        assert not reports[0].has_errors
        assert reports[1].has_errors

    def test_severity_labels(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Hello %s"\nmsgstr "你好"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        for err in report.errors:
            assert err.severity in (
                "critical",
                "functional",
                "cosmetic",
                "extraction",
                "other",
            )

    def test_custom_exclude(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Hello %s"\nmsgstr "你好"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN", exclude={"printf"})
        report = runner.check_unit(unit, index=1)
        checker_names = {e.checker for e in report.errors}
        assert "printf" not in checker_names

    def test_unchanged_translation_detected(self):
        from aitran.qa import QARunner

        po = _po('#: src/app.py:1\nmsgid "Hello"\nmsgstr "Hello"\n')
        unit = po.units[0]
        runner = QARunner(target_lang="zh_CN")
        report = runner.check_unit(unit, index=1)
        assert report.has_errors
        checker_names = {e.checker for e in report.errors}
        assert "unchanged" in checker_names
