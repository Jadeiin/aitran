"""Tests for translation and review workflow subagents."""

import json

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from aitran.agents import workflows
from aitran.toolsets._base import OrchestratorDeps


async def test_translation_workflow_subagent_calls_only_translation_tool(monkeypatch):
    calls = []

    async def fake_translate(ctx, **kwargs):  # noqa: RUF029
        calls.append((ctx.deps, kwargs))
        return "translated"

    monkeypatch.setattr(workflows, "translate_file", fake_translate)
    deps = OrchestratorDeps()
    agent = workflows.build_translation_workflow_agent(TestModel())
    await agent.run('{"path": "messages.po"}', deps=deps)

    assert len(calls) == 1
    assert calls[0][0] is deps
    assert set(calls[0][1]) == {
        "path",
        "source_lang",
        "target_lang",
        "output_path",
        "context_file",
        "batch_size",
        "jobs",
        "order",
        "profile",
    }


async def test_review_workflow_subagent_calls_only_review_tool(monkeypatch):
    calls = []

    async def fake_review(ctx, **kwargs):  # noqa: RUF029
        calls.append((ctx.deps, kwargs))
        return "reviewed"

    monkeypatch.setattr(workflows, "review_translated_file", fake_review)
    deps = OrchestratorDeps()
    agent = workflows.build_review_workflow_agent(TestModel())
    await agent.run(
        '{"input_file": "messages.translated.po", '
        '"output_file": "messages.reviewed.po"}',
        deps=deps,
    )

    assert len(calls) == 1
    assert calls[0][0] is deps
    assert calls[0][1]["path"] == "messages.translated.po"
    assert calls[0][1]["output_path"] == "messages.reviewed.po"
    assert set(calls[0][1]) == {
        "path",
        "source_lang",
        "target_lang",
        "output_path",
        "batch_size",
        "strict",
        "auto_fix",
    }


async def test_translation_workflow_preserves_delegated_output_file(monkeypatch):
    """The language model cannot silently drop an output-file task field."""
    calls = []

    async def fake_translate(_ctx, **kwargs):  # noqa: RUF029
        calls.append(kwargs)
        return f"Translated PO file: {kwargs['output_path']}"

    async def respond(messages, agent_info):  # noqa: RUF029
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("translated")])

        tool = next(
            tool for tool in agent_info.function_tools if tool.name == "run_translation"
        )
        required = tool.parameters_json_schema.get("required", [])
        # Reproduce the observed model behavior with the old schema: it maps
        # input_file to path but silently omits output_file.
        args = {"path": "collabora-ui-zh_CN.po"} if "path" in required else {}
        return ModelResponse(parts=[ToolCallPart("run_translation", args)])

    monkeypatch.setattr(workflows, "translate_file", fake_translate)
    agent = workflows.build_translation_workflow_agent(FunctionModel(function=respond))
    task = json.dumps({
        "input_file": "collabora-ui-zh_CN.po",
        "output_file": "collabora-ui-zh_CN.translated.po",
        "source_language": "en",
        "target_language": "zh_CN",
        "file_format": "po",
    })

    await agent.run(task, deps=OrchestratorDeps())

    assert calls[0]["path"] == "collabora-ui-zh_CN.po"
    assert calls[0]["output_path"] == "collabora-ui-zh_CN.translated.po"
