"""Tests for reviewer result constraints."""

import pytest
from pydantic import ValidationError

from aitran.agent_work import ReviewResult, ReviewTask, TranslationTask


def test_revise_requires_corrected_text():
    with pytest.raises(ValidationError, match="require corrected"):
        ReviewResult(index=1, verdict="revise")


def test_reject_can_require_human_retranslation():
    result = ReviewResult(index=1, verdict="reject", note="needs product context")
    assert result.corrected is None


def test_file_tasks_separate_profile_from_free_form_instructions():
    translation = TranslationTask.model_validate({
        "input_path": "messages.po",
        "output_path": "translated.po",
        "profile": "full",
        "instructions": "Use established mobile application terminology.",
    })
    review = ReviewTask.model_validate({
        "input_path": "translated.po",
        "output_path": "reviewed.po",
        "instructions": "Prefer concise UI wording.",
    })

    assert translation.profile == "full"
    assert translation.instructions.startswith("Use established")
    assert review.instructions == "Prefer concise UI wording."


def test_translation_task_recovers_prose_misplaced_in_profile():
    guidance = "软件/移动应用界面; 除非有通用译名, 否则保留产品名。"
    task = TranslationTask.model_validate({
        "input_path": "messages.po",
        "output_path": "translated.po",
        "profile": guidance,
    })

    assert task.profile == "full"
    assert task.instructions == guidance
