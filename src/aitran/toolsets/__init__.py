"""Toolsets for the orchestrator agent.

Each toolset wraps a domain-specific set of aitran functions as
pydantic-ai tools that the orchestrator agent can call.
"""

from aitran.toolsets.crowdin import crowdin_toolset
from aitran.toolsets.translate import translate_toolset
from aitran.toolsets.weblate import weblate_toolset

__all__ = [
    "crowdin_toolset",
    "translate_toolset",
    "weblate_toolset",
]
