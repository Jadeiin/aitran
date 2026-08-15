"""Contract tests for run-local translator and reviewer work."""

import json

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from aitran.agent_work import TranslationTask, _translation_state
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
