"""Shared types and helpers for orchestrator toolsets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai_backends import LocalBackend


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

    # File management backend
    backend: LocalBackend | None = None

    # Session persistence
    session_dir: Path = field(default_factory=lambda: Path(".aitran/sessions"))

    # Terminal reporter for tool outcome display (used by _reporting hooks).
    tool_reporter: Callable[[str, str, bool], None] | None = None


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
