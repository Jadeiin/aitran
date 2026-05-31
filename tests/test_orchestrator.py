"""Tests for orchestrator prompt construction."""

from aitran.agents.orchestrator import _build_orchestrator_system_prompt


def test_orchestrator_prompt_mentions_supported_formats_and_codes():
    prompt = _build_orchestrator_system_prompt()

    assert ".po" in prompt
    assert ".xliff" in prompt
    assert ".xlf" in prompt
    assert "Do not download JSON" in prompt
    assert "support is best" in prompt
    assert "When calling download tools" in prompt
    assert "Supported Translate Toolkit codes:" in prompt
    assert "zh_CN" in prompt
