"""Centralized tool outcome reporting via pydantic-ai Hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.capabilities.hooks import Hooks

if TYPE_CHECKING:
    from pydantic_ai import RunContext


def _report(ctx: RunContext, tool_name: str, message: str, ok: bool) -> None:
    """Forward to the terminal reporter if configured."""
    reporter = getattr(ctx.deps, "tool_reporter", None)
    if reporter is not None:
        reporter(tool_name, message, ok)


def build_reporting_hooks() -> Hooks:
    """Build Hooks that report tool outcomes to the terminal.

    Returns:
        Configured :class:`~pydantic_ai.capabilities.hooks.Hooks` instance.
    """
    hooks = Hooks()

    @hooks.on.after_tool_execute
    async def _report_success(  # noqa: RUF029
        ctx: RunContext,
        *,
        call,
        tool_def,  # noqa: ARG001
        args,  # noqa: ARG001
        result,
    ):
        _report(ctx, call.tool_name, str(result)[:200], ok=True)
        return result

    @hooks.on.tool_execute_error
    async def _report_error(  # noqa: RUF029
        ctx: RunContext,
        *,
        call,
        tool_def,  # noqa: ARG001
        args,  # noqa: ARG001
        error,
    ):
        msg = f"{type(error).__name__}: {error}"
        _report(ctx, call.tool_name, msg[:200], ok=False)
        return msg

    return hooks
