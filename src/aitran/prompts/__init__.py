"""Prompt loading, XML format/parse, and shared types for translation batches."""

import re
from importlib.resources import files
from typing import Protocol, runtime_checkable


@runtime_checkable
class UnitProtocol(Protocol):
    """Structural type for translation units from PO or XLIFF sources."""
    source: str
    context: str | None = None  # msgctxt (PO) or resname (XLIFF)
    comment: str | None = None  # extracted comments (PO) or notes (XLIFF)


def load_system_prompt() -> str:
    """Read the system prompt from package resources."""
    return (
        files("aitran.prompts")
        .joinpath("system.txt")
        .read_text("utf-8")
        .strip()
    )


def load_user_prompt() -> str:
    """Read the user guidelines from package resources."""
    return (
        files("aitran.prompts")
        .joinpath("user.txt")
        .read_text("utf-8")
        .strip()
    )


def format_batch_xml(units: list[UnitProtocol], start_index: int) -> str:
    """
    Build an XML string for one batch of translation units.

    Each unit is wrapped as:
      <translate index="N" context="...">source text</translate>
    or (without context):
      <translate index="N">source text</translate>
    """
    lines: list[str] = []
    for i, unit in enumerate(units):
        idx = start_index + i
        ctx = getattr(unit, "context", None)
        comment = getattr(unit, "comment", None)

        if ctx and comment:
            lines.append(
                f'<translate index="{idx}" context="{ctx}" comment="{comment}">'
                f"{unit.source}</translate>"
            )
        elif ctx:
            lines.append(
                f'<translate index="{idx}" context="{ctx}">'
                f"{unit.source}</translate>"
            )
        else:
            lines.append(
                f'<translate index="{idx}">{unit.source}</translate>'
            )
    return "\n".join(lines)


_XLATED_RE = re.compile(
    r'<translated index="(\d+)"[^>]*>(.*?)</translated>', re.DOTALL
)


def parse_translations(
    response_text: str, start_index: int, count: int
) -> list[str]:
    """
    Extract translations from the LLM response.

    Each translation should be wrapped as:
      <translated index="N">translated text</translated>

    Returns a list of translated strings in the same order as input units.
    """
    matches = _XLATED_RE.findall(response_text)
    indexed: dict[int, str] = {}
    for idx_str, text in matches:
        indexed[int(idx_str)] = text

    result: list[str] = []
    for i in range(count):
        idx = start_index + i
        if idx in indexed:
            result.append(indexed[idx])
        else:
            raise ParseError(
                f"Missing translation for index {idx} in LLM response"
            )

    return result


class ParseError(Exception):
    """Raised when the LLM response cannot be parsed."""
    pass


class StreamParser:
    """Incrementally parse <translated> tags from a streaming LLM response.

    Call feed(chunk) with each text chunk as it arrives. After each feed,
    check .newly_completed for (index, text) pairs that were just completed.
    """

    def __init__(self, start_index: int, count: int):
        self.start_index = start_index
        self.end_index = start_index + count - 1
        self._buffer = ""
        self._found: dict[int, str] = {}
        self.newly_completed: list[tuple[int, str]] = []

    @property
    def completed_count(self) -> int:
        return len(self._found)

    def feed(self, chunk: str) -> None:
        """Feed a new text chunk. Updates .newly_completed with any new matches."""
        self.newly_completed.clear()
        self._buffer += chunk
        for m in _XLATED_RE.finditer(self._buffer):
            idx = int(m.group(1))
            text = m.group(2)
            if idx not in self._found:
                self._found[idx] = text
                self.newly_completed.append((idx, text))

    def get_result(self) -> list[str]:
        """Return translations in order. Raises ParseError if any missing."""
        result: list[str] = []
        for i in range(self.start_index, self.end_index + 1):
            if i in self._found:
                result.append(self._found[i])
            else:
                raise ParseError(
                    f"Missing translation for index {i} in LLM response"
                )
        return result
