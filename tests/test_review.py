"""Tests for the review pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from translate.storage import po

from aitran.review import _run_review_async
from aitran.translate import PoTranslator

if TYPE_CHECKING:
    from pathlib import Path


def _po(content: str) -> po.pofile:
    return po.pofile.parsestring(content.encode())


async def test_run_review_saves_output_when_all_units_are_clean(tmp_path: Path):
    pofile = _po('#: src/a.py:1\nmsgid "Hello"\nmsgstr "你好"\n')
    output_path = tmp_path / "reviewed.po"

    summary = await _run_review_async(
        store=pofile,
        units=[pofile.units[0]],
        source_lang="en",
        target_lang="zh_CN",
        model_spec="openai:gpt-4o-mini",
        translator=PoTranslator(),
        output_path=str(output_path),
        batch_size=100,
        api_key="test-key",
    )

    assert summary == {"pass": 1, "revise": 0, "reject": 0}
    assert output_path.exists()
    assert 'msgstr "你好"' in output_path.read_text()
