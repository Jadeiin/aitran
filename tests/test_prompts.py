"""Tests for prompt loading."""

from aitran.prompts import load_system_prompt, load_user_prompt


def test_load_prompts_nonempty():
    sys_prompt = load_system_prompt()
    user_prompt = load_user_prompt()
    assert len(sys_prompt.strip()) > 0
    assert len(user_prompt.strip()) > 0


def test_user_prompt_describes_format():
    user_prompt = load_user_prompt()
    assert "translate-batch" in user_prompt
    assert "fuzzy" in user_prompt
    assert "note" in user_prompt


def test_user_prompt_defines_xml_escaping_contract():
    user_prompt = load_user_prompt()
    assert "XML escaping only as a transport encoding" in user_prompt
    assert "final text exactly as it should be saved" in user_prompt
    assert "Preserve literal escaped strings" in user_prompt
