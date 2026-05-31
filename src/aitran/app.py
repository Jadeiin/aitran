"""Interactive app — session management and deferred-tool run loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai.messages import ModelRequest, ModelResponse

from aitran.agents._base import build_model, fmt_base_url
from aitran.agents.orchestrator import (
    build_orchestrator_agent,
)
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from pydantic_ai.messages import ModelMessage
    from rich.console import Console, RenderableType
    from rich.live import Live


# Type for the approval callback.  Receives tool name and args, returns
# True to approve, False to deny, or a denial reason string.
# May be sync or async (prompt_toolkit prompts are async).
ApprovalCallback = Callable[[str, dict], bool | str | Awaitable[bool | str]]
SlashCommandResult = str
APP_PROMPT = "aitran> "
APPROVE_PROMPT = "Approve? [Y/n] "
REASON_PROMPT = "Reason (optional): "

_SLASH_COMMAND_HELP = {
    "/help": "Show available REPL commands.",
    "/approve on": "Enable automatic approval for approval-gated tools.",
    "/approve off": "Disable automatic approval.",
    "/approve status": "Show whether automatic approval is enabled.",
    "/new": "Start a new session (discard current conversation).",
    "/resume": "List saved sessions and restore one.",
    "/resume <id>": "Restore a specific session by ID.",
    "/exit": "Exit the app REPL.",
    "/quit": "Exit the app REPL.",
}

_SLASH_COMMANDS = list(_SLASH_COMMAND_HELP)


class _AppAutoSuggest(AutoSuggestFromHistory):
    """Auto-suggest slash commands in addition to history."""

    def __init__(self) -> None:
        super().__init__()
        self._commands = _SLASH_COMMANDS

    def get_suggestion(self, buffer: Buffer, document: Document) -> Suggestion | None:
        suggestion = super().get_suggestion(buffer, document)
        text = document.text_before_cursor.strip()
        for cmd in self._commands:
            if cmd.startswith(text) and cmd != text:
                return Suggestion(cmd[len(text) :])
        return suggestion


@dataclass
class _InteractiveTerminal:
    """Coordinate terminal prompts with Rich live rendering."""

    console: Console
    auto_approve: bool = False
    current_live: Live | None = None
    current_output: str = ""
    persisted_output: str = ""
    _live_paused_for_prompt: bool = False
    session: PromptSession[Any] | None = field(default=None, init=False)
    _ephemeral_session: PromptSession[Any] | None = field(
        default=None, init=False, repr=False
    )

    def init_prompt_session(self, history_path: Path) -> None:
        """Create the prompt_toolkit session with persistent history."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, InMemoryHistory

        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.touch(exist_ok=True)
        self.session = PromptSession(history=FileHistory(str(history_path)))
        self._ephemeral_session = PromptSession(history=InMemoryHistory())

    def begin_turn(self) -> None:
        """Reset per-turn render state before a new agent response streams."""
        self.current_output = ""
        self.persisted_output = ""
        self._live_paused_for_prompt = False

    def clear_screen(self) -> None:
        """Clear the terminal and reset all output state."""
        self.current_output = ""
        self.persisted_output = ""
        self.console.clear()

    async def approval(self, tool_name: str, args: dict) -> bool | str:
        """Prompt for tool approval without fighting live terminal rendering.

        Returns:
            True to approve, False or a denial reason to reject.
        """
        if self.auto_approve:
            self.report_tool_result(tool_name, "Auto-approved.", True)
            return True
        paused_here = self._pause_live_output()
        try:
            args_str = json.dumps(args, ensure_ascii=False, indent=2)
            self.console.print(f"\n[bold yellow]Approve:[/] [cyan]{tool_name}[/]")
            self.console.print(f"  {args_str}")
            answer = await self._read_line(APPROVE_PROMPT, record_history=False) or ""
            if answer.strip().lower() in ("n", "no"):
                reason = await self._read_line(REASON_PROMPT, record_history=False)
                if reason is None:
                    return False
                return reason.strip() or False
            return True
        finally:
            if paused_here:
                self._resume_live_output()

    def report_tool_result(self, tool_name: str, message: str, ok: bool) -> None:
        """Print an immediate status line for a completed approved tool."""
        paused_here = self._pause_live_output()
        try:
            status = "ok" if ok else "failed"
            style = "green" if ok else "red"
            self.console.print(
                f"[bold {style}]Tool {status}:[/] [cyan]{tool_name}[/] {message}"
            )
        finally:
            if paused_here:
                self._resume_live_output()

    async def follow_up(self, session_id: str) -> str | None:
        """Prompt for the next conversational turn.

        Returns:
            The user's reply, or None when the session should end.
        """
        self.console.print(
            "\n[dim]"
            f"Session {session_id}. Reply to continue, or press Enter to exit. "
            "Use /help to list REPL commands."
            "[/dim]"
        )
        return await self.prompt(APP_PROMPT)

    def handle_slash_command(self, text: str) -> SlashCommandResult:
        """Handle REPL slash commands.

        Returns:
            ``"handled"`` when consumed locally, ``"exit"`` when the REPL
            should terminate, or ``"unhandled"`` otherwise.
        """
        command = text.strip()
        if not command.startswith("/"):
            return "unhandled"

        if command in {"/exit", "/quit"}:
            self.console.print("[dim]Exiting app.[/dim]")
            return "exit"
        if command == "/help":
            self.console.print("[dim]Available REPL commands:[/dim]")
            for name, description in _SLASH_COMMAND_HELP.items():
                self.console.print(f"  [cyan]{name}[/] — {description}")
            return "handled"
        if not command.startswith("/approve"):
            self.console.print(
                f"[yellow]Unknown command:[/] {command}\n"
                "[dim]Use /help to list REPL commands.[/dim]"
            )
            return "handled"

        parts = command.split()
        action = parts[1].lower() if len(parts) > 1 else "toggle"
        if action == "on":
            self.auto_approve = True
        elif action == "off":
            self.auto_approve = False
        elif action == "toggle":
            self.auto_approve = not self.auto_approve
        elif action != "status":
            self.console.print(
                "[yellow]Usage:[/] /approve on | /approve off | /approve status"
            )
            return "handled"

        status = "on" if self.auto_approve else "off"
        self.console.print(f"[dim]Auto-approve is {status}.[/dim]")
        return "handled"

    async def handle_resume(
        self,
        command: str,
        session_dir: Path,
    ) -> tuple[list[ModelMessage], str] | None:
        """Handle ``/resume`` — list sessions and load one.

        Returns:
            ``(messages, session_id)`` for the selected session, or None if
            the user cancelled.
        """
        from datetime import datetime

        parts = command.split(maxsplit=1)
        target_id = parts[1].strip() if len(parts) > 1 else None

        if target_id:
            try:
                session = load_session(target_id, base=session_dir)
            except FileNotFoundError:
                self.console.print(f"[yellow]Session not found:[/] {target_id}")
                return None
            return self._finish_resume(target_id, session)

        entries = list_sessions(base=session_dir)
        if not entries:
            self.console.print("[dim]No saved sessions found.[/dim]")
            return None

        self.console.print("[dim]Saved sessions:[/dim]")
        for idx, (sid, mtime, count) in enumerate(entries, 1):
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006
            self.console.print(f"  [cyan]{idx}[/]  {sid}  {ts}  ({count} messages)")

        choice = await self._read_line("Session number: ")
        if choice is None:
            return None
        try:
            idx = int(choice)
            sid, _, _ = entries[idx - 1]
        except (ValueError, IndexError):
            self.console.print("[yellow]Invalid selection.[/yellow]")
            return None

        session = load_session(sid, base=session_dir)
        return self._finish_resume(sid, session)

    def _finish_resume(
        self, sid: str, session: Session
    ) -> tuple[list[ModelMessage], str]:
        """Print resume confirmation and replay history.

        Returns:
            Messages and session ID tuple.
        """
        self.console.print(
            f"[dim]Resumed session {sid} ({len(session.messages)} messages).[/dim]"
        )
        self._replay_messages(session.messages)
        return session.messages, sid

    def _replay_messages(self, messages: list[ModelMessage]) -> None:
        """Print conversation history to the terminal."""
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if part.part_kind == "user-prompt":
                        self.console.print(f"\n[bold cyan]> [/]{part.content}")
            elif isinstance(msg, ModelResponse):
                text = msg.text
                if text:
                    self.console.print(text)

    async def prompt(self, prompt_text: str) -> str | None:
        """Read a line from the user with history and auto-suggest.

        Uses prompt_toolkit's ``prompt_async`` when a session is available,
        falling back to Rich ``console.input`` otherwise.  Pauses live
        rendering during input to avoid terminal conflicts.

        Returns:
            The stripped user input, or None when the prompt is cancelled.
        """
        return await self._read_line(prompt_text, record_history=True)

    async def _read_line(
        self, prompt_text: str, *, record_history: bool = True
    ) -> str | None:
        """Read a single line, optionally recording to persistent history.

        Returns:
            The stripped user input, or None when the prompt is cancelled.
        """
        was_paused_here = self._pause_live_output()
        try:
            if self.session is not None:
                if record_history:
                    reply = await self.session.prompt_async(
                        prompt_text, auto_suggest=_AppAutoSuggest()
                    )
                else:
                    session = self._ephemeral_session or self.session
                    reply = await session.prompt_async(prompt_text)
            else:
                reply = self.console.input(prompt_text)
            reply = reply.strip()
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            if was_paused_here:
                self._resume_live_output()
        return reply or None

    def update_output(self, content: str) -> RenderableType:
        """Store the latest streamed output and return only the unpersisted suffix.

        Returns:
            Renderable content representing the portion not yet printed permanently.
        """
        from rich.markdown import Markdown
        from rich.text import Text

        self.current_output = content
        suffix = self._suffix_after_persisted(content)
        if not suffix:
            return Text("")
        return Markdown(suffix)

    def persist_live_output(self) -> None:
        """Permanently print the current streamed output if it has new content."""
        from rich.markdown import Markdown

        suffix = self._suffix_after_persisted(self.current_output)
        if suffix:
            self.console.print(Markdown(suffix))
            self.persisted_output = self.current_output

    def _suffix_after_persisted(self, content: str) -> str:
        """Return the portion of *content* not yet written permanently."""
        if self.persisted_output and content.startswith(self.persisted_output):
            return content[len(self.persisted_output) :]
        return content

    def _pause_live_output(self) -> bool:
        """Pause live rendering and persist visible output if needed.

        Returns:
            True when this call paused the live display and should resume it later.
        """
        live = self.current_live
        if self._live_paused_for_prompt or not live or not live.is_started:
            return False
        live.stop()
        self.persist_live_output()
        self._live_paused_for_prompt = True
        return True

    def _resume_live_output(self) -> None:
        """Resume live rendering after a prompt-driven pause."""
        live = self.current_live
        if not self._live_paused_for_prompt or not live:
            return
        live.start(refresh=True)
        self._live_paused_for_prompt = False


@dataclass
class Session:
    """A persisted orchestrator conversation."""

    session_id: str
    messages: list[ModelMessage]
    path: Path


def _session_dir(base: Path | None = None) -> Path:
    d = base or Path(".aitran/sessions")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(
    session: Session,
    *,
    base: Path | None = None,
) -> Path:
    """Persist a session's message history to disk.

    Returns:
        Path to the saved session file.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    d = _session_dir(base)
    path = d / f"{session.session_id}.json"
    data = ModelMessagesTypeAdapter.dump_json(session.messages)
    path.write_bytes(data)
    return path


def load_session(
    session_id: str,
    *,
    base: Path | None = None,
) -> Session:
    """Load a session from disk.

    Returns:
        Restored session.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    d = _session_dir(base)
    path = d / f"{session_id}.json"
    data = path.read_bytes()
    messages = ModelMessagesTypeAdapter.validate_json(data)
    return Session(session_id=session_id, messages=messages, path=path)


def list_sessions(
    *,
    base: Path | None = None,
) -> list[tuple[str, float, int]]:
    """List persisted sessions sorted by modification time (newest first).

    Returns:
        List of ``(session_id, mtime, message_count)`` tuples.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    d = _session_dir(base)
    pairs: list[tuple[Path, float]] = []
    for path in d.glob("*.json"):
        st = path.stat()
        pairs.append((path, st.st_mtime))
    pairs.sort(key=lambda t: t[1], reverse=True)

    results: list[tuple[str, float, int]] = []
    for path, mtime in pairs:
        sid = path.stem
        try:
            messages = ModelMessagesTypeAdapter.validate_json(path.read_bytes())
        except Exception:  # noqa: BLE001, S112
            continue
        results.append((sid, mtime, len(messages)))
    return results


def _cli_approval(tool_name: str, args: dict) -> bool | str:
    """Default CLI approval callback — prompts user on stdin.

    Returns:
        True to approve, False or a reason string to deny.
    """
    from rich.console import Console

    console = Console()
    args_str = json.dumps(args, ensure_ascii=False, indent=2)
    console.print(f"\n[bold yellow]Approve:[/] [cyan]{tool_name}[/]")
    console.print(f"  {args_str}")
    answer = input(APPROVE_PROMPT).strip().lower()
    if answer in ("n", "no"):
        reason = input(REASON_PROMPT).strip()
        return reason or False
    return True


def _build_deferred_handler(
    on_approval: ApprovalCallback,
) -> Callable[
    [RunContext[OrchestratorDeps], DeferredToolRequests],
    DeferredToolResults | Awaitable[DeferredToolResults | None] | None,
]:
    """Build a HandleDeferredToolCalls handler from an approval callback.

    The returned handler is async so it can ``await`` an async approval
    callback (e.g. prompt_toolkit prompts).  Sync callbacks are called
    directly without ``await``.

    Returns:
        Deferred tool call handler function.
    """

    async def handler(
        _ctx: RunContext[OrchestratorDeps], requests: DeferredToolRequests
    ) -> DeferredToolResults:
        from pydantic_ai import ToolApproved

        approvals: dict[str, bool | ToolApproved | ToolDenied] = {}

        for call in requests.approvals:
            args = call.args_as_dict()
            result = on_approval(call.tool_name, args)
            if inspect.isawaitable(result):
                result = await result
            if result is True:
                approvals[call.tool_call_id] = ToolApproved()
            elif result is False:
                approvals[call.tool_call_id] = ToolDenied()
            else:
                approvals[call.tool_call_id] = ToolDenied(str(result))

        # Deferred calls are auto-approved (external execution)
        calls: dict[str, object] = {
            call.tool_call_id: f"(executed {call.tool_name})" for call in requests.calls
        }

        return requests.build_results(approvals=approvals, calls=calls)

    return handler


async def run_app_async(
    prompt: str | None,
    *,
    orchestrator_model: str,
    orchestrator_api_key: str | None = None,
    orchestrator_api_host: str | None = None,
    orchestrator_temperature: float = 0.5,
    deps: OrchestratorDeps | None = None,
    session_id: str | None = None,
    resume: bool = False,
    auto_approve: bool = False,
    on_approval: ApprovalCallback | None = None,
    console: Console | None = None,
) -> str:
    """Run the interactive app with deferred-tool approval.

    Args:
        prompt: Optional initial natural-language request.
        orchestrator_model: Model spec for the orchestrator agent.
        orchestrator_api_key: API key for the orchestrator model.
        orchestrator_api_host: Custom API base URL for the orchestrator model.
        orchestrator_temperature: LLM temperature for the orchestrator model.
        deps: Orchestrator dependencies.
        session_id: Session ID to resume.
        resume: Whether to resume from a saved session.
        auto_approve: Whether approvals should be granted automatically.
        on_approval: Callback for tool approval decisions.
        console: Rich Console for streaming text output.

    Returns:
        Final text output from the app.
    """
    deps = deps or OrchestratorDeps()
    terminal = (
        _InteractiveTerminal(console, auto_approve=auto_approve)
        if console is not None and sys.stdin.isatty()
        else None
    )
    if terminal is not None:
        terminal.init_prompt_session(deps.session_dir / "prompt-history.txt")
    on_approval = on_approval or (
        terminal.approval if terminal is not None else _cli_approval
    )
    if terminal is not None:
        deps.tool_reporter = terminal.report_tool_result

    model = build_model(
        orchestrator_model,
        api_key=orchestrator_api_key,
        base_url=fmt_base_url(orchestrator_api_host),
        temperature=orchestrator_temperature,
    )

    handler = _build_deferred_handler(on_approval)
    agent = build_orchestrator_agent(model, deferred_handler=handler)

    # Restore or create session
    messages: list[ModelMessage] = []
    if resume and session_id:
        try:
            session = load_session(session_id, base=deps.session_dir)
            messages = session.messages
        except FileNotFoundError:
            pass

    sid = session_id or uuid.uuid4().hex[:12]

    interactive = terminal is not None
    next_prompt = prompt.strip() if prompt is not None else None
    final_output = ""

    if next_prompt is None and not interactive:
        return final_output

    while True:
        if next_prompt is None:
            assert terminal is not None
            next_prompt = await terminal.prompt(APP_PROMPT)
            if next_prompt is None:
                return final_output

        # Handle /resume and /new before sync slash commands (modify session state).
        cmd = next_prompt.strip()
        if terminal is not None and cmd == "/new":
            messages = []
            sid = uuid.uuid4().hex[:12]
            terminal.clear_screen()
            terminal.console.print(f"[dim]New session {sid}.[/dim]")
            next_prompt = None
            continue

        if terminal is not None and (cmd == "/resume" or cmd.startswith("/resume ")):
            result = await terminal.handle_resume(next_prompt, deps.session_dir)
            if result is not None:
                messages, sid = result
            next_prompt = None
            continue

        command_result = (
            terminal.handle_slash_command(next_prompt)
            if terminal is not None
            else "unhandled"
        )
        if command_result == "exit":
            return final_output
        if command_result == "handled":
            next_prompt = None
            continue

        if console is not None:
            all_msgs = await _run_streaming(
                agent,
                next_prompt,
                messages,
                deps,
                console,
                terminal=terminal,
            )
        else:
            result = await agent.run(next_prompt, deps=deps, message_history=messages)
            all_msgs = result.all_messages()

        messages = all_msgs
        final_output = _extract_output(all_msgs)

        # Persist after every turn so `--resume` can continue from the latest reply.
        session = Session(
            session_id=sid,
            messages=all_msgs,
            path=deps.session_dir / f"{sid}.json",
        )
        save_session(session, base=deps.session_dir)

        if not interactive:
            return final_output

        assert terminal is not None
        follow_up = await terminal.follow_up(sid)
        if follow_up is None:
            return final_output
        next_prompt = follow_up


async def _run_streaming(
    agent: Agent[OrchestratorDeps, str | DeferredToolRequests],
    prompt: str,
    messages: list[ModelMessage],
    deps: OrchestratorDeps,
    console: Console,
    *,
    terminal: _InteractiveTerminal | None = None,
) -> list[ModelMessage]:
    """Run the agent with live-streamed text output.

    Uses agent.iter() to stream text deltas through a rich.Live display,
    following the same pattern as pydantic-ai's own CLI.

    Returns:
        Complete message history after the run.
    """
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.status import Status
    from rich.text import Text

    status = Status("[dim]Working…[/dim]", console=console)
    with status, ExitStack() as stack:
        async with agent.iter(prompt, deps=deps, message_history=messages) as agent_run:
            if terminal is not None:
                terminal.begin_turn()
            live = Live(
                Text(""),
                refresh_per_second=15,
                console=console,
                transient=True,
                vertical_overflow="ellipsis",
            )
            final_renderable = Text("")
            saw_output = False
            try:
                if terminal is not None:
                    terminal.current_live = live
                async for node in agent_run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(agent_run.ctx) as handle_stream:
                            status.stop()
                            stack.enter_context(live)
                            async for content in handle_stream.stream_output(
                                debounce_by=None,
                            ):
                                if isinstance(content, DeferredToolRequests):
                                    continue
                                content_text = str(content)
                                final_renderable = (
                                    terminal.update_output(content_text)
                                    if terminal is not None
                                    else Markdown(content_text)
                                )
                                saw_output = True
                                live.update(final_renderable)
            finally:
                if terminal is not None:
                    terminal.current_live = None

        assert agent_run.result is not None
        if saw_output:
            if terminal is not None:
                terminal.persist_live_output()
            else:
                console.print(final_renderable)
        return agent_run.result.all_messages()


def _extract_output(messages: list[ModelMessage]) -> str:
    """Extract the final text output from the last assistant message.

    Returns:
        Text content of the last ModelResponse, or empty string.
    """
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            text = msg.text
            if text:
                return text
    return ""


def run_app(
    prompt: str | None,
    *,
    orchestrator_model: str,
    orchestrator_api_key: str | None = None,
    orchestrator_api_host: str | None = None,
    orchestrator_temperature: float = 0.5,
    deps: OrchestratorDeps | None = None,
    session_id: str | None = None,
    resume: bool = False,
    auto_approve: bool = False,
    on_approval: ApprovalCallback | None = None,
    console: Console | None = None,
) -> str:
    """Run the interactive app synchronously.

    Returns:
        Final text output from the app.
    """
    return asyncio.run(
        run_app_async(
            prompt,
            orchestrator_model=orchestrator_model,
            orchestrator_api_key=orchestrator_api_key,
            orchestrator_api_host=orchestrator_api_host,
            orchestrator_temperature=orchestrator_temperature,
            deps=deps,
            session_id=session_id,
            resume=resume,
            auto_approve=auto_approve,
            on_approval=on_approval,
            console=console,
        )
    )
