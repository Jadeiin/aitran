"""Contract tests for run-local translator and reviewer work."""

import json

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from aitran.agent_work import TranslationTask, _translation_state
from aitran.agents.orchestrator import build_orchestrator_agent
from aitran.agents.reviewer import build_review_file_agent
from aitran.agents.translator import build_translation_file_agent
from aitran.toolsets._base import OrchestratorDeps


def _write_po(path, entries):
    body = (
        'msgid ""\nmsgstr ""\n'
        '"Language: zh_CN\\n"\n'
        '"Plural-Forms: nplurals=1; plural=0;\\n"\n\n'
    )
    for source, target in entries:
        body += f'msgid "{source}"\nmsgstr "{target}"\n\n'
    path.write_text(body, encoding="utf-8")


def test_pending_translation_ignores_obsolete_untranslated_units(tmp_path):
    input_path = tmp_path / "messages.po"
    _write_po(input_path, [("Hello", "")])
    with input_path.open("a", encoding="utf-8") as po_file:
        po_file.write('#~ msgid "Removed"\n#~ msgstr ""\n')

    state = _translation_state(
        TranslationTask(
            input_path=str(input_path),
            output_path=str(tmp_path / "translated.po"),
            target_language="zh_CN",
        )
    )

    assert [str(unit.source) for _, unit in state.items] == ["Hello"]


@pytest.mark.parametrize("operation", ["translation", "review"])
@pytest.mark.parametrize("plural", [False, True])
@pytest.mark.parametrize("delegated", [False, True])
async def test_file_agents_receive_and_use_dictionary_terms(
    operation, plural, delegated, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *_args, **_kwargs: str(tmp_path)
    )
    config_dir = tmp_path / ".aitran"
    config_dir.mkdir()
    source_term = "People" if plural else "Hello"
    (config_dir / "dictionary.json").write_text(
        json.dumps({source_term: "术语译文", "Unused": "未用术语"}), encoding="utf-8"
    )
    input_path = tmp_path / "messages.po"
    output_path = tmp_path / "result.po"
    current_target = "" if operation == "translation" else "你好"
    if plural:
        _write_po(input_path, [])
        with input_path.open("a", encoding="utf-8") as po_file:
            po_file.write(
                f'msgid "Person"\nmsgid_plural "People"\nmsgstr[0] "{current_target}"\n'
            )
    else:
        _write_po(input_path, [("Hello", current_target)])

    async def respond(messages, agent_info):  # noqa: RUF029
        tools = {tool.name for tool in agent_info.function_tools}
        if f"get_{operation}_batch" in tools:
            return ModelResponse(parts=[ToolCallPart(f"get_{operation}_batch", {})])
        if f"submit_{operation}_batch" in tools:
            page = next(
                part.content
                for message in reversed(messages)
                for part in reversed(message.parts)
                if isinstance(part, ToolReturnPart)
                and part.tool_name == f"get_{operation}_batch"
            )
            glossary = getattr(page, "glossary", [])
            assert glossary == [{"source": source_term, "target": "术语译文"}]
            target = glossary[0]["target"]
            if operation == "translation":
                results = {
                    "translations": [
                        {"index": page.units[0].index, "targets": [target]}
                    ]
                }
            else:
                results = {
                    "reviews": [
                        {
                            "index": page.units[0].index,
                            "verdict": "revise",
                            "corrected": target,
                        }
                    ]
                }
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        f"submit_{operation}_batch",
                        {"batch_id": page.batch_id, **results},
                    )
                ]
            )
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == f"finish_{operation}"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(f"finish_{operation}", {})])

    task = json.dumps({
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_language": "zh_CN",
    })
    worker_model = FunctionModel(function=respond)
    deps = OrchestratorDeps()
    if delegated:

        async def orchestrate(messages, _agent_info):  # noqa: RUF029
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
                        {
                            "agent_name": (
                                "translator"
                                if operation == "translation"
                                else "reviewer"
                            ),
                            "task": task,
                        },
                    )
                ]
            )

        agent = build_orchestrator_agent(
            FunctionModel(function=orchestrate), worker_model=worker_model
        )
        deferred = await agent.run("Process this file", deps=deps)
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
    else:
        builder = (
            build_translation_file_agent
            if operation == "translation"
            else build_review_file_agent
        )
        await builder(worker_model).run(task, deps=deps)

    target_field = "msgstr[0]" if plural else "msgstr"
    assert f'{target_field} "术语译文"' in output_path.read_text(encoding="utf-8")


async def test_translator_pages_file_and_returns_exact_artifact(tmp_path):
    input_path = tmp_path / "messages.po"
    output_path = tmp_path / "messages.translated.po"
    _write_po(input_path, [("Hello", ""), ("World", "")])
    seen_requests: list[str] = []

    async def respond(messages, agent_info):  # noqa: RUF029
        seen_requests.append(str(messages))
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
            units = page.units
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "submit_translation_batch",
                        {
                            "batch_id": page.batch_id,
                            "translations": [
                                {
                                    "index": unit.index,
                                    "targets": [f"译-{unit.sources[0]}"],
                                }
                                for unit in units
                            ],
                        },
                    )
                ]
            )
        if "finish_translation" in tools:
            return ModelResponse(parts=[ToolCallPart("finish_translation", {})])
        return ModelResponse(parts=[TextPart("unexpected")])

    agent = build_translation_file_agent(FunctionModel(function=respond))
    task = json.dumps({
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_language": "zh_CN",
        "max_page_units": 1,
        "max_page_chars": 4096,
    })
    result = await agent.run(task, deps=OrchestratorDeps())

    artifact = json.loads(result.output)
    assert artifact["output_path"] == str(output_path)
    assert artifact["processed_units"] == 2
    saved = output_path.read_text(encoding="utf-8")
    assert "译-Hello" in saved
    assert "译-World" in saved
    # Completed pages remain in history so the provider can reuse a stable prefix.
    page_two_request = next(
        request
        for request in seen_requests
        if "World" in request and "译-World" not in request
    )
    assert "Hello" in page_two_request


async def test_reviewer_checks_every_page_and_applies_sparse_corrections(tmp_path):
    input_path = tmp_path / "translated.po"
    output_path = tmp_path / "reviewed.po"
    _write_po(input_path, [("Hello", "你好"), ("World", "世界。")])

    async def respond(messages, agent_info):  # noqa: RUF029
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == "finish_review"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("done")])
        tools = {tool.name for tool in agent_info.function_tools}
        if "get_review_batch" in tools:
            return ModelResponse(parts=[ToolCallPart("get_review_batch", {})])
        if "submit_review_batch" in tools:
            page = next(
                part.content
                for message in reversed(messages)
                for part in reversed(message.parts)
                if isinstance(part, ToolReturnPart)
                and part.tool_name == "get_review_batch"
            )
            unit = page.units[0]
            reviews = []
            if unit.sources == ["World"]:
                reviews = [
                    {
                        "index": unit.index,
                        "verdict": "revise",
                        "corrected": "世界",
                        "note": "remove unnecessary punctuation",
                    }
                ]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "submit_review_batch",
                        {"batch_id": page.batch_id, "reviews": reviews},
                    )
                ]
            )
        if "finish_review" in tools:
            return ModelResponse(parts=[ToolCallPart("finish_review", {})])
        return ModelResponse(parts=[TextPart("unexpected")])

    agent = build_review_file_agent(FunctionModel(function=respond))
    result = await agent.run(
        json.dumps({
            "input_path": str(input_path),
            "output_path": str(output_path),
            "target_language": "zh_CN",
            "max_page_units": 1,
        }),
        deps=OrchestratorDeps(),
    )

    artifact = json.loads(result.output)
    assert artifact["output_path"] == str(output_path)
    assert artifact["processed_units"] == 2
    assert artifact["summary"] == {"pass": 1, "revise": 1, "reject": 0}
    assert 'msgstr "世界"' in output_path.read_text(encoding="utf-8")
