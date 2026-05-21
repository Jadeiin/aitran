"""Tests for the multi-turn TranslationSession."""

from unittest.mock import MagicMock, patch

from aitran.prompts import UnitProtocol
from aitran.translate import TranslationSession


class FakeUnit:
    def __init__(self, source: str, context: str | None = None):
        self.source = source
        self.context = context
        self.comment = None


def test_session_setup_populates_messages():
    session = TranslationSession(model="gpt-4o-mini", timeout=30000)
    session.setup(
        system_prompt="You are a translator.",
        user_prompt="Translate from en to zh.",
        source_lang="en",
        target_lang="zh",
        context="This is context.",
        dict_entries=[("hello", "你好")],
    )

    assert len(session.messages) == 5
    assert session.messages[0]["role"] == "system"
    assert "This is context." in session.messages[0]["content"]
    assert session.messages[1]["role"] == "user"
    assert "en" in session.messages[1]["content"]
    assert session.messages[2]["role"] == "assistant"
    assert session.messages[3]["role"] == "user"
    assert "hello" in session.messages[3]["content"]
    assert session.messages[4]["role"] == "assistant"
    assert "你好" in session.messages[4]["content"]


def test_session_setup_without_dict():
    session = TranslationSession(model="gpt-4o-mini")
    session.setup(
        system_prompt="Sys",
        user_prompt="User",
        source_lang="en",
        target_lang="fr",
    )
    # Should only have system + user + assistant (no dict entries)
    assert len(session.messages) == 3


@patch("aitran.translate.litellm.completion")
def test_translate_batch_calls_api(mock_completion):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='<translated index="1">你好</translated>\n<translated index="2">世界</translated>'
            )
        )
    ]
    mock_completion.return_value = mock_response

    session = TranslationSession(model="gpt-4o-mini")
    session.setup(
        system_prompt="Sys",
        user_prompt="User",
        source_lang="en",
        target_lang="zh",
    )

    units = [FakeUnit("hello"), FakeUnit("world")]
    result = session.translate_batch(units)

    assert result == ["你好", "世界"]
    assert mock_completion.called

    # Check that messages grew correctly
    # Initial: [system, user, assistant] = 3
    # After batch: + [user, assistant] = 5
    assert len(session.messages) == 5
    assert session.messages[3]["role"] == "user"
    assert "hello" in session.messages[3]["content"]
    assert session.messages[4]["role"] == "assistant"


@patch("aitran.translate.litellm.completion")
def test_multi_batch_accumulates_messages(mock_completion):
    """Multiple batches add to the same conversation."""
    def make_response(content):
        r = MagicMock()
        r.choices = [MagicMock(message=MagicMock(content=content))]
        return r

    mock_completion.side_effect = [
        make_response('<translated index="1">你好</translated>'),
        make_response('<translated index="2">世界</translated>'),
    ]

    session = TranslationSession(model="gpt-4o-mini")
    session.setup(
        system_prompt="Sys",
        user_prompt="User",
        source_lang="en",
        target_lang="zh",
    )

    session.translate_batch([FakeUnit("hello")])
    session.translate_batch([FakeUnit("world")])

    # Initial: [sys, user, asst] = 3
    # Batch 1: + [user, asst] = 5
    # Batch 2: + [user, asst] = 7
    assert len(session.messages) == 7

    # System prompt is only in messages[0], never repeated
    sys_count = sum(1 for m in session.messages if m["role"] == "system")
    assert sys_count == 1
