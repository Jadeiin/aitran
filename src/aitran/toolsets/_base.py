"""Shared types and helpers for orchestrator toolsets."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ToolReporter = Callable[[str, str, bool], None]


@dataclass
class OrchestratorDeps:
    """Dependencies injected into the orchestrator agent.

    Credentials for platforms are optional — only the ones relevant to
    the user's request need to be filled in.
    """

    # Crowdin credentials
    crowdin_token: str | None = None
    crowdin_organization: str | None = None
    crowdin_base_url: str | None = None
    crowdin_timeout: int = 120

    # Weblate credentials
    weblate_url: str | None = None
    weblate_token: str | None = None

    # Model config for translate/review sub-tasks
    translate_model: str = "deepseek:deepseek-v4-flash"
    translate_api_key: str | None = None
    translate_api_host: str | None = None
    translate_temperature: float = 0.1

    # Session persistence
    session_dir: Path = field(default_factory=lambda: Path(".aitran/sessions"))

    # Optional terminal reporter for approved tool completion.
    tool_reporter: ToolReporter | None = None


def summarize_list(items: list[dict], *, label: str, name_field: str = "name") -> str:
    """Format a list of API results as a concise summary for the LLM.

    Returns:
        JSON-formatted summary string.
    """
    if not items:
        return f"No {label} found."
    summary = []
    for item in items:
        entry = {"id": item.get("id")}
        name = item.get(name_field) or item.get("name") or item.get("path")
        if name:
            entry[name_field] = name
        # Include a few commonly useful fields
        for key in ("status", "approval", "translated", "phrases"):
            if key in item:
                entry[key] = item[key]
        summary.append(entry)
    return json.dumps(summary, ensure_ascii=False, indent=2)


def summarize_progress(items: list[dict]) -> str:
    """Format Crowdin progress data as a concise summary.

    Returns:
        JSON-formatted progress summary.
    """
    if not items:
        return "No progress data."
    summary = []
    for item in items:
        entry: dict = {}
        data = item.get("data", item)
        if "name" in data:
            entry["name"] = data["name"]
        if "language" in data:
            lang = data["language"]
            entry["language"] = lang.get("name") or lang.get("id", "")
        progress_keys = (
            "translationProgress",
            "approvalProgress",
            "phrases",
            "translated",
            "approved",
        )
        for key in progress_keys:
            if key in data:
                entry[key] = data[key]
        summary.append(entry)
    return json.dumps(summary, ensure_ascii=False, indent=2)


def error_message(operation: str, error: Exception) -> str:
    """Format an error as a concise message for the LLM.

    Returns:
        Error summary string.
    """
    return f"{operation} failed: {type(error).__name__}: {error}"


def report_tool_outcome(
    deps: OrchestratorDeps,
    *,
    tool_name: str,
    message: str,
    ok: bool,
) -> None:
    """Emit an immediate tool completion update when a reporter is configured."""
    if deps.tool_reporter is not None:
        deps.tool_reporter(tool_name, message, ok)
