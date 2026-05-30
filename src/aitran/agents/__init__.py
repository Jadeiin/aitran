"""Agent definitions for aitran.

Each agent lives in its own module. Shared utilities (model routing and prompt
field helpers) live in ``_base``.

Note: orchestrator agent is imported lazily (not re-exported here) to avoid
circular imports with the toolsets module.
"""

from aitran.agents._base import (
    build_model,
    build_retrying_http_client,
    fmt_base_url,
    format_language_label,
    safe_prompt_text,
)
from aitran.agents.reviewer import (
    SYSTEM_PROMPT as REVIEWER_SYSTEM_PROMPT,
)
from aitran.agents.reviewer import (
    USER_PROMPT as REVIEWER_USER_PROMPT,
)
from aitran.agents.reviewer import (
    ReviewBatch,
    ReviewDeps,
    ReviewedUnit,
    build_review_input_xml,
    build_reviewer_agent,
)
from aitran.agents.translator import (
    SYSTEM_PROMPT as TRANSLATOR_SYSTEM_PROMPT,
)
from aitran.agents.translator import (
    USER_PROMPT as TRANSLATOR_USER_PROMPT,
)
from aitran.agents.translator import (
    TranslatedUnit,
    TranslationBatch,
    TranslationDeps,
    build_translation_input_xml,
    build_translator_agent,
)

__all__ = [
    "REVIEWER_SYSTEM_PROMPT",
    "REVIEWER_USER_PROMPT",
    "TRANSLATOR_SYSTEM_PROMPT",
    "TRANSLATOR_USER_PROMPT",
    "ReviewBatch",
    "ReviewDeps",
    "ReviewedUnit",
    "TranslatedUnit",
    "TranslationBatch",
    "TranslationDeps",
    "build_model",
    "build_retrying_http_client",
    "build_review_input_xml",
    "build_reviewer_agent",
    "build_translation_input_xml",
    "build_translator_agent",
    "fmt_base_url",
    "format_language_label",
    "safe_prompt_text",
]
