"""Tests for the review pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.retries import AsyncTenacityTransport

from aitran.agents import ReviewBatch, ReviewedUnit
from aitran.review import _run_review_async, build_default_reviewer
from aitran.translate import PoTranslator
from tests.helpers import po_parse as _po

if TYPE_CHECKING:
    from pathlib import Path


def test_default_reviewer_uses_retrying_http_transport():
    agent = build_default_reviewer("openai:gpt-4o-mini", api_key="test-key")
    provider = agent.model.__dict__["_provider"]
    openai_client = provider.__dict__["_client"]

    assert isinstance(openai_client._client._transport, AsyncTenacityTransport)


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

    assert summary == {"pass": 1, "revise": 0, "reject": 0, "skip": 0}
    assert output_path.exists()
    assert 'msgstr "你好"' in output_path.read_text()


async def test_review_preserves_completed_chunks_after_later_failure(
    monkeypatch, tmp_path: Path
):
    pofile = _po(
        '#: src/a.py:1\nmsgid "Hello"\nmsgstr "您好"\n\n'
        '#: src/b.py:1\nmsgid "World"\nmsgstr "世界"\n'
    )
    output_path = tmp_path / "reviewed.po"
    calls = 0

    async def stream_fn(_messages, _agent_info):  # noqa: RUF029
        nonlocal calls
        calls += 1
        if calls == 1:
            yield (
                '{"units":[{"index":1,"verdict":"revise",'
                '"corrected":null,"note":"too formal"}]}'
            )
            return
        raise RuntimeError("later review chunk failed")

    monkeypatch.setattr(
        "aitran.review.build_model",
        lambda *_args, **_kwargs: FunctionModel(stream_function=stream_fn),
    )

    with pytest.raises(RuntimeError, match="later review chunk failed"):
        await _run_review_async(
            store=pofile,
            units=list(pofile.units),
            source_lang="en",
            target_lang="zh_CN",
            model_spec="openai:gpt-4o-mini",
            translator=PoTranslator(),
            output_path=str(output_path),
            batch_size=1,
            strict=True,
        )

    out = output_path.read_text(encoding="utf-8")
    assert 'msgid "Hello"' in out
    assert "#, fuzzy" in out
    assert "(review) revise: too formal" in out
    assert 'msgid "World"' in out
    assert 'msgstr "世界"' in out


async def test_review_progress_updates_on_streamed_units(monkeypatch, tmp_path: Path):
    pofile = _po(
        '#: src/a.py:1\nmsgid "Hello"\nmsgstr "您好"\n\n'
        '#: src/b.py:1\nmsgid "World"\nmsgstr "世界"\n'
    )

    class FakeRun:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def stream_output(self, **_kwargs):
            yield ReviewBatch(
                units=[
                    ReviewedUnit(
                        index=1,
                        verdict="revise",
                        corrected=None,
                        note="too formal",
                    )
                ]
            )

        async def get_output(self):
            return ReviewBatch(
                units=[
                    ReviewedUnit(
                        index=1,
                        verdict="revise",
                        corrected=None,
                        note="too formal",
                    )
                ]
            )

        def new_messages(self) -> list:
            return []

    class FakeReviewerAgent:
        def run_stream(self, *_args, **_kwargs):
            return FakeRun()

    class FakeProgress:
        console = None

        def __init__(self) -> None:
            self.updates: list[dict] = []

        def add_task(self, *_args, **_kwargs):
            return 1

        def update(self, _task_id, **kwargs):
            self.updates.append(kwargs)

    monkeypatch.setattr("aitran.review.build_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "aitran.review.build_reviewer_agent",
        lambda *_args, **_kwargs: FakeReviewerAgent(),
    )

    progress = FakeProgress()
    await _run_review_async(
        store=pofile,
        units=list(pofile.units),
        source_lang="en",
        target_lang="zh_CN",
        model_spec="openai:gpt-4o-mini",
        translator=PoTranslator(),
        output_path=str(tmp_path / "reviewed.po"),
        batch_size=2,
        strict=True,
        progress=progress,
    )

    assert progress.updates == [{"completed": 1}, {"completed": 2}]
