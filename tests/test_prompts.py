"""Tests for prompt constants."""

from aitran.agents.translator import SYSTEM_PROMPT, USER_PROMPT


def test_prompts_nonempty():
    assert len(SYSTEM_PROMPT.strip()) > 0
    assert len(USER_PROMPT.strip()) > 0


def test_user_prompt_describes_format():
    assert "translate-batch" in USER_PROMPT
    assert "fuzzy" in USER_PROMPT
    assert "note" in USER_PROMPT
