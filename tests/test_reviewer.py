"""Tests for the reviewer agent."""

from pydantic_ai.models.test import TestModel
from translate.storage.po import pofile

from aitran.agents.reviewer import (
    ReviewBatch,
    ReviewDeps,
    ReviewedUnit,
    build_review_input_xml,
    build_reviewer_agent,
)
from aitran.qa import UnitQAReport


def _po(content: str) -> pofile:
    return pofile.parsestring(content.encode())


class TestReviewedUnit:
    def test_revise_with_correction(self):
        unit = ReviewedUnit(
            index=1,
            verdict="revise",
            corrected="你好 %s",
            note="missing placeholder",
        )
        assert unit.corrected == "你好 %s"

    def test_reject_without_correction(self):
        unit = ReviewedUnit(
            index=1, verdict="reject", note="meaning is completely wrong"
        )
        assert unit.corrected is None


class TestReviewBatch:
    def test_batch_roundtrip(self):
        batch = ReviewBatch(
            units=[
                ReviewedUnit(index=2, verdict="revise", corrected="fixed"),
                ReviewedUnit(index=3, verdict="reject"),
            ]
        )
        assert len(batch.units) == 2
        assert batch.units[0].verdict == "revise"
        assert batch.units[0].corrected == "fixed"


class TestBuildReviewerAgent:
    def test_agent_name(self):
        model = TestModel()
        agent = build_reviewer_agent(model)
        assert agent.name == "aitran-reviewer"

    def test_agent_runs_with_test_model(self):
        model = TestModel(
            custom_output_args={
                "units": [
                    {"index": 2, "verdict": "revise", "corrected": "fix"},
                    {"index": 3, "verdict": "reject", "note": "bad"},
                ]
            }
        )
        agent = build_reviewer_agent(model)
        deps = ReviewDeps(
            source_lang="en",
            target_lang="zh_CN",
            context="UI strings",
            expected_indices=(1, 2, 3),
        )
        result = agent.run_sync("review these", deps=deps)
        assert len(result.output.units) == 2
        assert result.output.units[0].verdict == "revise"
        assert result.output.units[0].corrected == "fix"


class TestBuildReviewInputXml:
    def test_plural_units_include_all_source_and_target_forms(self):
        po = _po(
            "#: src/a.py:1\n"
            'msgid "apple %d"\n'
            'msgid_plural "apples %d"\n'
            'msgstr[0] "苹果 %d"\n'
            'msgstr[1] "苹果们"\n'
        )
        xml = build_review_input_xml([po.units[0]], [UnitQAReport(index=1)])

        assert "apple %d" in xml
        assert "apples %d" in xml
        assert "苹果 %d" in xml
        assert "苹果们" in xml
        assert "sources" in xml
        assert "targets" in xml
