"""Tests for orchestrator prompt construction."""

from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from aitran.agents.orchestrator import (
    _build_orchestrator_system_prompt,
    build_orchestrator_agent,
)
from aitran.toolsets._base import OrchestratorDeps


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


async def test_orchestrator_exposes_approved_harness_delegation_tool():
    tool_kinds = {}

    async def respond(_messages, agent_info):  # noqa: RUF029
        tool_kinds.update({tool.name: tool.kind for tool in agent_info.function_tools})
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task",
                    {
                        "agent_name": "translator",
                        "task": '{"path": "messages.po", "target_lang": "zh_CN"}',
                    },
                )
            ]
        )

    agent = build_orchestrator_agent(
        FunctionModel(function=respond),
        worker_model=TestModel(),
    )
    result = await agent.run("Translate a file", deps=OrchestratorDeps())

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ["delegate_task"]
    assert tool_kinds["delegate_task"] == "function"
    assert "delegate_translation" not in tool_kinds
    assert "delegate_review" not in tool_kinds
    assert "translate_file" not in tool_kinds
    assert "review_translated_file" not in tool_kinds


async def test_approved_harness_delegation_forwards_parent_deps(monkeypatch):
    from aitran.agents import workflows

    received_deps = []

    async def fake_translate(ctx, **_kwargs):  # noqa: RUF029
        received_deps.append(ctx.deps)
        return "translated"

    async def respond(messages, _agent_info):  # noqa: RUF029
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "delegate_task"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task",
                    {"agent_name": "translator", "task": '{"path": "messages.po"}'},
                )
            ]
        )

    monkeypatch.setattr(workflows, "translate_file", fake_translate)
    deps = OrchestratorDeps()
    agent = build_orchestrator_agent(
        FunctionModel(function=respond),
        worker_model=TestModel(),
    )

    deferred = await agent.run("Translate a file", deps=deps)
    assert isinstance(deferred.output, DeferredToolRequests)
    approval = deferred.output.approvals[0]

    completed = await agent.run(
        deps=deps,
        message_history=deferred.all_messages(),
        deferred_tool_results=DeferredToolResults(
            approvals={approval.tool_call_id: True}
        ),
    )

    assert completed.output == "done"
    assert received_deps == [deps]
