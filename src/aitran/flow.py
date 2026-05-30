"""Orchestrator flow — session management and deferred-tool run loop."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)

from aitran.agents.orchestrator import (
    build_orchestrator_agent,
    build_orchestrator_model,
)
from aitran.toolsets._base import OrchestratorDeps

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from rich.console import Console


# Type for the approval callback.  Receives tool name and args, returns
# True to approve, False to deny, or a denial reason string.
ApprovalCallback = Callable[[str, dict], bool | str]


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
    answer = input("Approve? [Y/n] ").strip().lower()
    if answer in ("n", "no"):
        reason = input("Reason (optional): ").strip()
        return reason or False
    return True


def _build_deferred_handler(
    on_approval: ApprovalCallback,
) -> Callable[
    [RunContext[OrchestratorDeps], DeferredToolRequests], DeferredToolResults
]:
    """Build a HandleDeferredToolCalls handler from an approval callback.

    Returns:
        Deferred tool call handler function.
    """

    def handler(
        _ctx: RunContext[OrchestratorDeps], requests: DeferredToolRequests
    ) -> DeferredToolResults:
        from pydantic_ai import ToolApproved

        approvals: dict[str, bool | ToolApproved | ToolDenied] = {}

        for call in requests.approvals:
            args = call.args if isinstance(call.args, dict) else {}
            result = on_approval(call.tool_name, args)
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


async def run_flow(
    prompt: str,
    *,
    orchestrator_model: str | None = None,
    orchestrator_api_key: str | None = None,
    deps: OrchestratorDeps | None = None,
    session_id: str | None = None,
    resume: bool = False,
    on_approval: ApprovalCallback | None = None,
    console: Console | None = None,
) -> str:
    """Run the orchestrator flow with deferred-tool approval.

    Args:
        prompt: User's natural-language request.
        orchestrator_model: Model spec for the orchestrator agent.
        orchestrator_api_key: API key for the orchestrator model.
        deps: Orchestrator dependencies (credentials, config).
        session_id: Session ID to resume.
        resume: Whether to resume from a saved session.
        on_approval: Callback for tool approval decisions.
            Defaults to CLI stdin prompt.
        console: Rich Console for streaming text output.
            When provided, agent responses are streamed live.

    Returns:
        Final text output from the orchestrator.
    """
    deps = deps or OrchestratorDeps()
    on_approval = on_approval or _cli_approval

    model = build_orchestrator_model(
        orchestrator_model,
        api_key=orchestrator_api_key,
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

    if console is not None:
        all_msgs = await _run_streaming(agent, prompt, messages, deps, console)
    else:
        result = await agent.run(prompt, deps=deps, message_history=messages)
        all_msgs = result.all_messages()

    # Persist final state
    session = Session(
        session_id=sid,
        messages=all_msgs,
        path=deps.session_dir / f"{sid}.json",
    )
    save_session(session, base=deps.session_dir)

    return _extract_output(all_msgs)


async def _run_streaming(
    agent: Agent[OrchestratorDeps, str | DeferredToolRequests],
    prompt: str,
    messages: list[ModelMessage],
    deps: OrchestratorDeps,
    console: Console,
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

    status = Status("[dim]Working…[/dim]", console=console)
    with status, ExitStack() as stack:
        async with agent.iter(prompt, deps=deps, message_history=messages) as agent_run:
            live = Live(
                "",
                refresh_per_second=15,
                console=console,
                vertical_overflow="ellipsis",
            )
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
                            live.update(Markdown(str(content)))

        assert agent_run.result is not None
        return agent_run.result.all_messages()


def _extract_output(messages: list[ModelMessage]) -> str:
    """Extract the final text output from the last assistant message.

    Returns:
        Text content of the last ModelResponse, or empty string.
    """
    from pydantic_ai.messages import ModelResponse

    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            text = msg.text
            if text:
                return text
    return ""
