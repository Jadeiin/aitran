"""Tests for prompt loading and XML format/parse round-trip."""

from aitran.prompts import (
    ParseError,
    StreamParser,
    UnitProtocol,
    format_batch_xml,
    parse_translations,
)


class FakeUnit:
    def __init__(self, source: str, context: str | None = None, comment: str | None = None):
        self.source = source
        self.context = context
        self.comment = comment


def test_format_and_parse_round_trip():
    units = [
        FakeUnit("Hello", "greeting"),
        FakeUnit("Goodbye"),
        FakeUnit("File", "Menu"),
    ]
    xml = format_batch_xml(units, start_index=1)
    # Build a matching response
    response = """<translated index="1">你好</translated>
<translated index="2">再见</translated>
<translated index="3">文件</translated>"""
    result = parse_translations(response, start_index=1, count=3)
    assert result == ["你好", "再见", "文件"]


def test_parse_handles_out_of_order():
    units = [FakeUnit("A"), FakeUnit("B"), FakeUnit("C")]
    xml = format_batch_xml(units, start_index=5)
    response = """<translated index="7">Third</translated>
<translated index="5">First</translated>
<translated index="6">Second</translated>"""
    result = parse_translations(response, start_index=5, count=3)
    assert result == ["First", "Second", "Third"]


def test_parse_missing_index_raises():
    units = [FakeUnit("A"), FakeUnit("B")]
    response = '<translated index="1">Only one</translated>'
    try:
        parse_translations(response, start_index=1, count=2)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass


def test_format_includes_context():
    units = [FakeUnit("File", "Menu")]
    xml = format_batch_xml(units, start_index=1)
    assert 'context="Menu"' in xml
    assert "File" in xml
    assert 'index="1"' in xml


def test_format_no_context():
    units = [FakeUnit("Hello")]
    xml = format_batch_xml(units, start_index=3)
    assert "context=" not in xml
    assert "Hello" in xml
    assert 'index="3"' in xml


def test_parse_with_extra_text():
    """LLM sometimes adds explanatory text around the tags."""
    response = """Here are the translations:

<translated index="10">First</translated>
Some extra note here.
<translated index="11">Second</translated>"""
    result = parse_translations(response, start_index=10, count=2)
    assert result == ["First", "Second"]


def test_unitprotocol_structural_match():
    """Verify UnitProtocol matches objects with source attribute."""
    u = FakeUnit("test", "ctx")
    assert isinstance(u, UnitProtocol)


def test_stream_parser_incremental():
    """StreamParser finds completed tags as chunks arrive."""
    parser = StreamParser(start_index=1, count=2)
    parser.feed('<translated index="1">Hello</translated>')
    assert parser.completed_count == 1
    assert len(parser.newly_completed) == 1
    assert parser.newly_completed[0] == (1, "Hello")

    parser.feed('<translated index="2">World</translated>')
    assert parser.completed_count == 2
    assert parser.newly_completed[0] == (2, "World")

    assert parser.get_result() == ["Hello", "World"]


def test_stream_parser_partial():
    """StreamParser handles partial tags gracefully."""
    parser = StreamParser(start_index=1, count=2)
    parser.feed('<translated index="1">First part')
    assert parser.completed_count == 0  # no complete tag yet

    parser.feed(' of translation</translated>')
    assert parser.completed_count == 1
    assert parser.newly_completed[0][1] == "First part of translation"


def test_stream_parser_out_of_order():
    """StreamParser assigns translations by index, not arrival order."""
    parser = StreamParser(start_index=5, count=2)
    parser.feed('<translated index="6">Second</translated>')
    assert parser.completed_count == 1
    parser.feed('<translated index="5">First</translated>')
    assert parser.completed_count == 2
    assert parser.get_result() == ["First", "Second"]


def test_stream_parser_ignores_duplicates():
    """StreamParser ignores repeated indices."""
    parser = StreamParser(start_index=1, count=1)
    parser.feed('<translated index="1">A</translated>')
    assert parser.completed_count == 1
    parser.feed('<translated index="1">B</translated>')
    assert parser.completed_count == 1  # unchanged
    assert parser.get_result() == ["A"]
