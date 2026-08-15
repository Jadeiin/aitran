"""Tests for headless Agent event handling."""

from __future__ import annotations

import json

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from aitran import headless
from aitran.agents.translator import build_translation_file_agent
from aitran.headless import _run


def test_run_reports_each_submitted_translation_page(tmp_path):
    input_path = tmp_path / "messages.po"
    output_path = tmp_path / "translated.po"
    input_path.write_text(
        'msgid ""\nmsgstr ""\n'
        '"Language: zh_CN\\n"\n\n'
        'msgid "Hello"\nmsgstr ""\n\n'
        'msgid "World"\nmsgstr ""\n',
        encoding="utf-8",
    )

    def respond(messages, agent_info):
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "finish_translation"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("done")])
        tools = {tool.name for tool in agent_info.function_tools}
        if "get_translation_batch" in tools:
            return ModelResponse(parts=[ToolCallPart("get_translation_batch", {})])
        if "submit_translation_batch" in tools:
            page = next(
                part.content
                for message in reversed(messages)
                for part in reversed(message.parts)
                if isinstance(part, ToolReturnPart)
                and part.tool_name == "get_translation_batch"
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "submit_translation_batch",
                        {
                            "batch_id": page.batch_id,
                            "translations": [
                                {"index": unit.index, "targets": ["译文"]}
                                for unit in page.units
                            ],
                        },
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart("finish_translation", {})])

    async def stream(messages, agent_info):  # noqa: RUF029
        response = respond(messages, agent_info)
        part = response.parts[0]
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {
                0: DeltaToolCall(
                    name=part.tool_name,
                    json_args=json.dumps(part.args),
                    tool_call_id=part.tool_call_id,
                )
            }

    progress = []
    agent = build_translation_file_agent(FunctionModel(stream_function=stream))
    artifact = _run(
        agent,
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "target_language": "zh_CN",
            "max_page_units": 1,
        },
        on_progress=lambda path, update: progress.append((path, update)),
    )

    assert artifact["processed_units"] == 2
    assert [update.accepted for _, update in progress] == [0, 1, 2]
    assert [update.remaining for _, update in progress] == [2, 1, 0]
    assert {path for path, _ in progress} == {str(input_path)}


def test_translate_directory_creates_one_agent_per_file(monkeypatch, tmp_path):
    for name in ("a.po", "b.po"):
        (tmp_path / name).write_text("", encoding="utf-8")
    agents = []
    runs = []

    def fake_translation_agent(*_args):
        agent = object()
        agents.append(agent)
        return agent

    def fake_run(agent, task, *, on_progress):
        runs.append((agent, task["input_path"], on_progress))
        return {"output_path": task["output_path"]}

    monkeypatch.setattr(headless, "_translation_agent", fake_translation_agent)
    monkeypatch.setattr(headless, "_run", fake_run)

    artifacts = headless.translate_directory(
        model="test:model",
        dir_path=str(tmp_path),
        suffixes=(".po",),
        source_lang="en",
        target_lang="zh_CN",
        context_file=None,
        max_page_units=100,
        max_page_chars=4096,
        jobs=2,
    )

    assert len(agents) == 2
    assert {agent for agent, _, _ in runs} == set(agents)
    assert [artifact["output_path"] for artifact in artifacts] == [
        str(tmp_path / "a.po"),
        str(tmp_path / "b.po"),
    ]
