"""Tests for interactive app behavior."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from aitran import app
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pathlib import Path


PLAN_TEXT = "执行计划\n\n是否按此计划执行?"
DONE_TEXT = "已执行完成。"
APP_PROMPT = "aitran> "


def _noop_init_prompt_session(_self: object, *_args: object, **_kwargs: object) -> None:
    """Stub for init_prompt_session in tests that don't need prompt_toolkit."""


class DummyConsole:
    """Small console stub for interactive flow tests."""

    def __init__(self, replies: list[str]):
        self._replies = iter(replies)
        self.printed: list[str] = []
        self.prompts: list[str] = []

    def print(self, *args, **kwargs) -> None:
        del kwargs
        self.printed.append(" ".join(str(arg) for arg in args))

    def input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self._replies)


class DummyLive:
    """Small live stub for prompt pause/resume tests."""

    def __init__(self, started: bool = True):
        self.is_started = started
        self.events: list[str] = []

    def stop(self) -> None:
        self.events.append("stop")
        self.is_started = False

    def start(self, *, refresh: bool = False) -> None:
        self.events.append(f"start:{refresh}")
        self.is_started = True


def _response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


def _fake_builder(*_args, **_kwargs) -> object:
    return object()


async def test_run_app_continues_interactively(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, list[ModelResponse]]] = []
    outputs = [[_response(PLAN_TEXT)], [_response(DONE_TEXT)]]

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, deps, console, terminal
        await _tick()
        calls.append((prompt, list(messages)))
        return outputs.pop(0)

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        app._InteractiveTerminal, "init_prompt_session", _noop_init_prompt_session
    )

    console = DummyConsole(["继续执行", ""])
    deps = OrchestratorDeps(session_dir=tmp_path / "sessions")

    result = await app.run_app_async("先看看状态", deps=deps, console=console)

    assert result == DONE_TEXT
    assert [prompt for prompt, _ in calls] == ["先看看状态", "继续执行"]
    assert calls[0][1] == []
    assert calls[1][1][0].text == PLAN_TEXT
    assert console.prompts == [APP_PROMPT, APP_PROMPT]
    assert len(list(deps.session_dir.glob("*.json"))) == 1


async def test_run_app_stays_one_shot_without_tty(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, messages, deps, console, terminal
        await _tick()
        calls.append(prompt)
        return [_response(PLAN_TEXT)]

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: False)

    console = DummyConsole([])
    deps = OrchestratorDeps(session_dir=tmp_path / "sessions")

    result = await app.run_app_async("先看看状态", deps=deps, console=console)

    assert result == PLAN_TEXT
    assert calls == ["先看看状态"]
    assert console.prompts == []


async def test_run_app_enters_repl_when_prompt_missing(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, messages, deps, console, terminal
        await _tick()
        calls.append(prompt)
        return [_response(DONE_TEXT)]

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        app._InteractiveTerminal, "init_prompt_session", _noop_init_prompt_session
    )

    console = DummyConsole(["翻译这个组件", ""])
    deps = OrchestratorDeps(session_dir=tmp_path / "sessions")

    result = await app.run_app_async(None, deps=deps, console=console)

    assert result == DONE_TEXT
    assert calls == ["翻译这个组件"]
    assert console.prompts == [APP_PROMPT, APP_PROMPT]


async def test_run_app_handles_approve_slash_command(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, messages, deps, console, terminal
        await _tick()
        calls.append(prompt)
        return [_response(DONE_TEXT)]

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        app._InteractiveTerminal, "init_prompt_session", _noop_init_prompt_session
    )

    console = DummyConsole(["/approve on", "翻译这个组件", ""])
    deps = OrchestratorDeps(session_dir=tmp_path / "sessions")

    result = await app.run_app_async(None, deps=deps, console=console)

    assert result == DONE_TEXT
    assert calls == ["翻译这个组件"]
    assert "Auto-approve is on." in console.printed[0]


async def test_run_app_handles_exit_slash_command(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, messages, deps, console, terminal
        await _tick()
        calls.append(prompt)
        return [_response(DONE_TEXT)]

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        app._InteractiveTerminal, "init_prompt_session", _noop_init_prompt_session
    )

    console = DummyConsole(["/exit"])
    deps = OrchestratorDeps(session_dir=tmp_path / "sessions")

    result = await app.run_app_async(None, deps=deps, console=console)

    assert result == ""
    assert calls == []
    assert "Exiting app." in console.printed[0]


async def test_interactive_terminal_pauses_live_while_prompting():
    console = DummyConsole(["继续执行"])
    live = DummyLive()
    terminal = app._InteractiveTerminal(console=console, current_live=live)
    terminal.current_output = "当前模型输出"

    reply = await terminal.prompt(APP_PROMPT)

    assert reply == "继续执行"
    assert console.prompts == [APP_PROMPT]
    assert live.events == ["stop", "start:True"]
    assert live.is_started is True
    assert terminal.persisted_output == "当前模型输出"
    assert len(console.printed) == 1


async def test_interactive_terminal_approval_uses_shared_console():
    console = DummyConsole(["n", "参数不对"])
    terminal = app._InteractiveTerminal(console=console)

    result = await terminal.approval("translate_file", {"path": "demo.po"})

    assert result == "参数不对"
    assert console.prompts == [
        "Approve? [Y/n] ",
        "Reason (optional): ",
    ]


async def test_interactive_terminal_approval_preserves_output_order():
    console = DummyConsole([""])
    live = DummyLive()
    terminal = app._InteractiveTerminal(console=console, current_live=live)
    terminal.current_output = "先下载文件:"

    args = {"object_path": "demo/path"}
    await terminal.approval("weblate__download_translation", args)

    assert live.events == ["stop", "start:True"]
    assert terminal.persisted_output == "先下载文件:"
    assert len(console.printed) == 3
    assert "Approve:" in console.printed[1]
    assert "object_path" in console.printed[2]


def test_interactive_terminal_reports_completed_tool():
    console = DummyConsole([])
    live = DummyLive()
    terminal = app._InteractiveTerminal(console=console, current_live=live)
    terminal.current_output = "执行中"

    terminal.report_tool_result("translate_file", "Translated PO file: demo.po", True)

    assert live.events == ["stop", "start:True"]
    assert terminal.persisted_output == "执行中"
    assert "Tool ok:" in console.printed[1]
    assert "translate_file" in console.printed[1]


async def test_interactive_terminal_auto_approves_when_enabled():
    console = DummyConsole([])
    terminal = app._InteractiveTerminal(console=console, auto_approve=True)

    result = await terminal.approval("translate_file", {"path": "demo.po"})

    assert result is True
    assert "Auto-approved." in console.printed[0]


def test_interactive_terminal_handles_approve_status_command():
    console = DummyConsole([])
    terminal = app._InteractiveTerminal(console=console)

    handled = terminal.handle_slash_command("/approve status")

    assert handled == "handled"
    assert "Auto-approve is off." in console.printed[0]


def test_interactive_terminal_handles_help_command():
    console = DummyConsole([])
    terminal = app._InteractiveTerminal(console=console)

    result = terminal.handle_slash_command("/help")

    assert result == "handled"
    assert "Available REPL commands:" in console.printed[0]
    assert any("/exit" in line for line in console.printed)


def test_interactive_terminal_handles_unknown_command():
    console = DummyConsole([])
    terminal = app._InteractiveTerminal(console=console)

    result = terminal.handle_slash_command("/wat")

    assert result == "handled"
    assert "Unknown command:" in console.printed[0]


async def test_deferred_handler_parses_tool_args_from_json():
    seen: list[dict] = []

    def on_approval(tool_name: str, args: dict) -> bool:
        assert tool_name == "translate_file"
        seen.append(args)
        return True

    handler = app._build_deferred_handler(on_approval)
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart(
                tool_name="translate_file",
                args='{"path":"demo.po","target_lang":"zh_CN"}',
                tool_call_id="call-1",
            )
        ]
    )

    result = await handler(None, requests)

    assert seen == [{"path": "demo.po", "target_lang": "zh_CN"}]
    assert "call-1" in result.approvals


def test_list_sessions_returns_sorted_entries(tmp_path: Path):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    # Create two session files with different mtimes.
    old = session_dir / "aaa.json"
    old.write_bytes(ModelMessagesTypeAdapter.dump_json([]))
    time.sleep(0.05)
    new = session_dir / "bbb.json"
    new.write_bytes(ModelMessagesTypeAdapter.dump_json([]))

    entries = app.list_sessions(base=session_dir)

    assert len(entries) == 2
    assert entries[0][0] == "bbb"  # newest first
    assert entries[1][0] == "aaa"


def test_list_sessions_skips_corrupt_files(tmp_path: Path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "bad.json").write_text("not valid json")
    (session_dir / "empty.json").write_text("[]")

    entries = app.list_sessions(base=session_dir)

    # bad.json is skipped; empty.json has 0 messages but is valid.
    assert len(entries) == 1
    assert entries[0][0] == "empty"


async def test_run_app_handles_resume_by_id(monkeypatch, tmp_path: Path):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    calls: list[str] = []
    outputs = [[_response(DONE_TEXT)]]

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, messages, deps, console, terminal
        await _tick()
        calls.append(prompt)
        return outputs.pop(0)

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        app._InteractiveTerminal, "init_prompt_session", _noop_init_prompt_session
    )

    # Pre-create a session file.
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    saved_msgs = [_response(PLAN_TEXT)]
    (session_dir / "old123.json").write_bytes(
        ModelMessagesTypeAdapter.dump_json(saved_msgs)
    )

    console = DummyConsole(["/resume old123", "继续执行", ""])
    deps = OrchestratorDeps(session_dir=session_dir)

    result = await app.run_app_async(None, deps=deps, console=console)

    assert result == DONE_TEXT
    assert calls == ["继续执行"]
    assert any("Resumed session old123" in line for line in console.printed)
    assert any(PLAN_TEXT in line for line in console.printed)


async def test_run_app_handles_resume_selection(monkeypatch, tmp_path: Path):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    calls: list[str] = []
    outputs = [[_response(DONE_TEXT)]]

    async def fake_run_streaming(
        agent, prompt, messages, deps, console, *, terminal=None
    ):
        del agent, messages, deps, console, terminal
        await _tick()
        calls.append(prompt)
        return outputs.pop(0)

    monkeypatch.setattr(app, "build_orchestrator_model", _fake_builder)
    monkeypatch.setattr(app, "build_orchestrator_agent", _fake_builder)
    monkeypatch.setattr(app, "_run_streaming", fake_run_streaming)
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        app._InteractiveTerminal, "init_prompt_session", _noop_init_prompt_session
    )

    # Pre-create a session file.
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "abc.json").write_bytes(
        ModelMessagesTypeAdapter.dump_json([_response(PLAN_TEXT)])
    )

    # /resume lists sessions, user picks "1", then continues.
    console = DummyConsole(["/resume", "1", "继续执行", ""])
    deps = OrchestratorDeps(session_dir=session_dir)

    result = await app.run_app_async(None, deps=deps, console=console)

    assert result == DONE_TEXT
    assert calls == ["继续执行"]
    assert any("Saved sessions:" in line for line in console.printed)
    assert any("Resumed session abc" in line for line in console.printed)
    assert any(PLAN_TEXT in line for line in console.printed)


async def test_handle_resume_no_sessions(tmp_path: Path):
    console = DummyConsole([])
    terminal = app._InteractiveTerminal(console=console)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()

    result = await terminal.handle_resume("/resume", session_dir)

    assert result is None
    assert any("No saved sessions" in line for line in console.printed)


async def _tick() -> None:
    """Keep fake async hooks visibly asynchronous for linting."""
    await asyncio.sleep(0)
