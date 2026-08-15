"""Tests for specialized Agent instructions."""

from aitran.agents.reviewer import SYSTEM_PROMPT as REVIEW_PROMPT
from aitran.agents.reviewer import TASK_CONTRACT as REVIEW_TASK_CONTRACT
from aitran.agents.translator import SYSTEM_PROMPT as TRANSLATE_PROMPT
from aitran.agents.translator import TASK_CONTRACT as TRANSLATE_TASK_CONTRACT


def test_translator_prompt_requires_paginated_tools():
    assert "batch_id" in TRANSLATE_PROMPT
    assert "finish_translation" in TRANSLATE_PROMPT
    assert "plural-form" in TRANSLATE_PROMPT
    assert "placeholders" in TRANSLATE_PROMPT
    assert "leading and trailing whitespace" in TRANSLATE_PROMPT
    assert "Do not XML-escape" in TRANSLATE_PROMPT
    assert "fuzzy" in TRANSLATE_PROMPT
    assert "translator note" in TRANSLATE_PROMPT
    assert '"profile": "fast" | "full"' in TRANSLATE_TASK_CONTRACT
    assert '"instructions": string' in TRANSLATE_TASK_CONTRACT
    assert "free-form" in TRANSLATE_TASK_CONTRACT


def test_reviewer_prompt_is_full_and_sparse():
    assert "inspect every translation" in REVIEW_PROMPT
    assert "Omission means pass" in REVIEW_PROMPT
    assert "finish_review" in REVIEW_PROMPT
    assert "Critical" in REVIEW_PROMPT
    assert "false positive" in REVIEW_PROMPT
    assert "placeholders" in REVIEW_PROMPT
    assert "meaning accuracy" in REVIEW_PROMPT
    assert '"instructions": string' in REVIEW_TASK_CONTRACT
