"""Tests for headless Agent event handling."""

from __future__ import annotations

import json

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

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
