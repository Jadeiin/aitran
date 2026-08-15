"""Tests for format adapters retained behind the Agent toolset seam."""

import pytest
from translate.storage import po

from aitran.agent_work import TranslationResult
from aitran.agents import build_model
from aitran.translate import (
    PoTranslator,
    _decode_serialized_markup,
    _order_units,
)


def _po(text: str):
    return po.pofile.parsestring(text.encode())


def test_po_adapter_selects_empty_fuzzy_and_partial_plural():
    store = _po(
        'msgid ""\nmsgstr ""\n'
        '"Language: de\\n"\n'
        '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "empty"\nmsgstr ""\n\n'
        '#, fuzzy\nmsgid "fuzzy"\nmsgstr "alt"\n\n'
        'msgid "apple"\nmsgid_plural "apples"\n'
        'msgstr[0] "Apfel"\nmsgstr[1] ""\n\n'
        'msgid "done"\nmsgstr "fertig"\n'
    )

    assert [str(unit.source) for unit in PoTranslator.get_untranslated(store)] == [
        "empty",
        "fuzzy",
        "apple",
    ]


def test_po_adapter_applies_plural_and_fuzzy_results():
    store = _po(
        'msgid ""\nmsgstr ""\n'
        '"Language: de\\n"\n'
        '"Plural-Forms: nplurals=2; plural=(n != 1);\\n"\n\n'
        'msgid "apple"\nmsgid_plural "apples"\nmsgstr[0] ""\nmsgstr[1] ""\n'
    )
    unit = store.units[1]
    result = TranslationResult(
        index=1, targets=["Apfel", "Äpfel"], fuzzy=True, note="check"
    )

    PoTranslator.apply_batch(store, [unit], [result])

    assert list(unit.target.strings) == ["Apfel", "Äpfel"]
    assert unit.isfuzzy()
    assert "check" in unit.getnotes()


def test_decode_markup_only_reverses_entities_present_in_source():
    assert (
        _decode_serialized_markup("Use <code> & text", "使用 &lt;code&gt; &amp; 文本")
        == "使用 <code> & 文本"
    )
    assert (
        _decode_serialized_markup("Literal &lt;code&gt;", "保留 &lt;code&gt;")
        == "保留 &lt;code&gt;"
    )


def test_order_units_keeps_stable_ties():
    first = _po('msgid "b"\nmsgstr ""\n').units[0]
    second = _po('msgid "a"\nmsgstr ""\n').units[0]
    assert _order_units([first, second], "source") == [second, first]


def test_build_model_requires_provider_separator():
    with pytest.raises(ValueError, match="provider:model"):
        build_model("gpt")
