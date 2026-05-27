"""Tests for the Pydantic AI translator agent and file I/O adapters."""

import threading
import time
from dataclasses import dataclass
from importlib.metadata import version as package_version

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from translate.misc import xml_helpers
from translate.misc.multistring import multistring
from translate.storage import po

from aitran.agent import (
    TranslatedUnit,
    TranslationDeps,
    build_input_xml,
    build_model,
    build_translator_agent,
)
from aitran.translate import (
    PoTranslator,
    XliffTranslator,
    _translate_batch,
    translate_po,
    translate_po_dir,
    translate_xliff_dir,
)

DEFAULT_TEST_MODEL = "deepseek:deepseek-v4-flash"

# ── Helpers ──────────────────────────────────────────────────────


@dataclass
class FakeUnit:
    source: str
    context: str | None = None
    _note: str | None = None

    def getcontext(self) -> str:
        return self.context or ""

    def getnotes(self) -> str:
        return self._note or ""


def _make_deps(expected_indices=(1, 2), plural_tags=None):
    return TranslationDeps(
        source_lang="en",
        target_lang="zh_CN",
        context="",
        dict_entries=[],
        expected_indices=expected_indices,
        plural_tags=plural_tags,
    )


# ── build_input_xml ──────────────────────────────────────────────


def test_build_input_xml_basic():
    units = [FakeUnit("Hello"), FakeUnit("World")]
    xml = build_input_xml(units, start_index=1)
    assert "<translate-batch>" in xml
    assert "</translate-batch>" in xml
    assert "<index>1</index>" in xml and "<source>Hello</source>" in xml
    assert "<index>2</index>" in xml and "<source>World</source>" in xml


def test_build_input_xml_with_context():
    units = [FakeUnit("File", context="Menu", _note="top-level")]
    xml = build_input_xml(units, start_index=5)
    assert "<index>5</index>" in xml
    assert "<context>Menu</context>" in xml
    assert "<note>top-level</note>" in xml


def test_build_input_xml_omits_none_fields():
    units = [FakeUnit("Plain")]
    xml = build_input_xml(units, start_index=1)
    assert "context" not in xml
    assert "comment" not in xml
    assert "null" not in xml


def test_build_input_xml_strips_invalid_xml_characters():
    units = [FakeUnit("Hello \x08 world")]
    xml = build_input_xml(units, start_index=1)
    assert "\x08" not in xml
    assert "Hello  world" in xml
    xml_helpers.parse_xml(xml)


def test_build_input_xml_preserves_escaped_markup_after_sanitizing():
    units = [
        FakeUnit(
            'Click <a href="/docs?a=1&b=2">docs</a><br/><code>x & y</code>\x08',
            context="HTML label <strong>primary</strong>",
            _note="Keep <code>, <a>, and <br/> tags.",
        )
    ]
    xml = build_input_xml(units, start_index=1)

    assert "\x08" not in xml
    assert '&lt;a href="/docs?a=1&amp;b=2"&gt;docs&lt;/a&gt;' in xml
    assert "&lt;br/&gt;" in xml
    assert "&lt;code&gt;x &amp; y&lt;/code&gt;" in xml
    assert "&lt;strong&gt;primary&lt;/strong&gt;" in xml
    xml_helpers.parse_xml(xml)


# ── Translate batch via TestModel ─────────────────────────────────


async def test_translate_batch_success():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "targets": ["你好"], "fuzzy": False, "note": None},
                {"index": 2, "targets": ["世界"], "fuzzy": False, "note": None},
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("hello"), FakeUnit("world")]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps(),
            [],
            on_progress=None,
        )
    assert len(results) == 2
    assert results[0].targets[0] == "你好"
    assert results[1].targets[0] == "世界"


async def test_translate_batch_fuzzy_flag():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "targets": ["x"], "fuzzy": True, "note": "unsure"},
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("ambiguous")]
    with agent.override(model=model):
        results = await _translate_batch(
            agent, units, 1, _make_deps((1,)), [], on_progress=None
        )
    assert results[0].fuzzy is True
    assert results[0].note == "unsure"


# ── ModelRetry via FunctionModel ──────────────────────────────────


async def test_translate_batch_retries_on_missing_index():
    """FunctionModel returns incomplete on call 1, complete on call 2."""

    call_count = 0

    async def stream_fn(_messages, _agent_info):  # noqa: RUF029
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ('{"translations":[{"index":1,"targets":["ok"],"fuzzy":false}]}')
        else:
            yield (
                '{"translations":['
                '{"index":1,"targets":["ok"],"fuzzy":false},'
                '{"index":2,"targets":["yes"],"fuzzy":false}'
                "]}"
            )

    fn_model = FunctionModel(stream_function=stream_fn)
    agent = build_translator_agent(fn_model)
    units = [FakeUnit("a"), FakeUnit("b")]
    with agent.override(model=fn_model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1, 2)),
            [],
            on_progress=None,
        )
    assert len(results) == 2
    assert call_count >= 2  # should have retried at least once


# ── Message history ───────────────────────────────────────────────


async def test_message_history_accumulates():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "targets": ["你好"], "fuzzy": False, "note": None},
            ],
        }
    )
    agent = build_translator_agent(model)
    units_a = [FakeUnit("hello")]
    units_b = [FakeUnit("world")]

    history: list = []
    with agent.override(model=model):
        await _translate_batch(
            agent,
            units_a,
            1,
            _make_deps((1,)),
            history,
            on_progress=None,
        )
        # Second call: continue the same conversation
        model2 = TestModel(
            custom_output_args={
                "translations": [
                    {"index": 2, "targets": ["世界"], "fuzzy": False, "note": None},
                ],
            }
        )
        agent2 = build_translator_agent(model2)
        with agent2.override(model=model2):
            await _translate_batch(
                agent2,
                units_b,
                2,
                _make_deps((2,)),
                history,
                on_progress=None,
            )

    # History should contain messages from both batches
    assert len(history) > 2


# ── PoTranslator apply_batch ──────────────────────────────────────


def test_po_apply_batch_writes_fuzzy():
    pf = po.pofile()
    u = po.pounit(source="hello")
    pf.addunit(u)
    PoTranslator.apply_batch(
        pf,
        [u],
        [
            TranslatedUnit(index=1, targets=["你好"], fuzzy=True, note="check me"),
        ],
    )
    out = bytes(pf).decode()
    assert "#, fuzzy" in out
    assert "check me" in out
    assert "msgstr" in out


def test_po_apply_batch_writes_clean():
    pf = po.pofile()
    u = po.pounit(source="hello")
    u.markfuzzy(True)  # was fuzzy before
    pf.addunit(u)
    PoTranslator.apply_batch(
        pf,
        [u],
        [
            TranslatedUnit(index=1, targets=["你好"], fuzzy=False),
        ],
    )
    out = bytes(pf).decode()
    # PO header always has "#, fuzzy" — check only the unit block
    assert 'msgid "hello"' in out
    # After apply clears fuzzy, the unit should not have its own "#, fuzzy" line
    lines = out.split("\n")
    unit_start = next(i for i, line in enumerate(lines) if 'msgid "hello"' in line)
    unit_block = "\n".join(lines[unit_start:])
    assert "#, fuzzy" not in unit_block


def test_po_apply_batch_no_note_no_comment():
    pf = po.pofile()
    u = po.pounit(source="hello")
    pf.addunit(u)
    PoTranslator.apply_batch(
        pf,
        [u],
        [
            TranslatedUnit(index=1, targets=["你好"], fuzzy=False),
        ],
    )
    out = bytes(pf).decode()
    assert 'msgstr "你好"' in out


def test_po_apply_batch_syncs_plural_count_from_target_language():
    pf = po.pofile()
    pf.updateheader(add=True, Language="en")
    u = po.pounit(source=multistring(["file", "files"]))
    pf.addunit(u)
    PoTranslator.apply_batch(
        pf,
        [u],
        [
            TranslatedUnit(index=1, targets=["file"], fuzzy=False),
        ],
    )
    out = bytes(pf).decode()
    assert 'msgstr[0] "file"' in out
    assert 'msgstr[1] ""' in out


def test_po_apply_batch_uses_single_plural_form_for_chinese():
    pf = po.pofile()
    pf.updateheader(add=True, Language="zh_CN")
    u = po.pounit(source=multistring(["file", "files"]))
    pf.addunit(u)
    PoTranslator.apply_batch(
        pf,
        [u],
        [
            TranslatedUnit(index=1, targets=["文件"], fuzzy=False),
        ],
    )
    out = bytes(pf).decode()
    assert pf.get_plural_tags() == ["other"]
    assert 'msgstr[0] "文件"' in out
    assert "msgstr[1]" not in out


def test_translate_po_infers_target_language_from_header(monkeypatch, tmp_path):
    source = tmp_path / "messages.po"
    source.write_text(
        (
            'msgid ""\n'
            'msgstr ""\n'
            '"Language: zh_CN\\n"\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgid "Hello"\n'
            'msgstr ""\n'
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_translation(**kwargs):
        captured["target_lang"] = kwargs["target_lang"]

    monkeypatch.setattr("aitran.translate._run_translation", fake_run_translation)

    translate_po(
        model=DEFAULT_TEST_MODEL,
        po_path=str(source),
        source_lang="en",
        target_lang="",
        verbose=False,
        output_path=str(source),
        context_file=None,
        context_length=4096,
    )

    assert captured["target_lang"] == "zh_CN"


def test_translate_po_infers_legacy_target_language_from_script_header(
    monkeypatch, tmp_path
):
    source = tmp_path / "messages.po"
    source.write_text(
        (
            'msgid ""\n'
            'msgstr ""\n'
            '"Language: zh_Hans\\n"\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgid "Hello"\n'
            'msgstr ""\n'
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_translation(**kwargs):
        captured["target_lang"] = kwargs["target_lang"]

    monkeypatch.setattr("aitran.translate._run_translation", fake_run_translation)

    translate_po(
        model=DEFAULT_TEST_MODEL,
        po_path=str(source),
        source_lang="en",
        target_lang="",
        verbose=False,
        output_path=str(source),
        context_file=None,
        context_length=4096,
    )

    assert captured["target_lang"] == "zh_CN"


def test_translate_po_infers_target_language_from_language_team(monkeypatch, tmp_path):
    source = tmp_path / "messages.po"
    source.write_text(
        (
            'msgid ""\n'
            'msgstr ""\n'
            '"Language-Team: French <traduc@traduc.org>\\n"\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgid "Hello"\n'
            'msgstr ""\n'
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_translation(**kwargs):
        captured["target_lang"] = kwargs["target_lang"]
        kwargs["translator"].save(kwargs["store"], kwargs["output_path"])

    monkeypatch.setattr("aitran.translate._run_translation", fake_run_translation)

    translate_po(
        model=DEFAULT_TEST_MODEL,
        po_path=str(source),
        source_lang="en",
        target_lang="",
        verbose=False,
        output_path=str(source),
        context_file=None,
        context_length=4096,
    )

    out = source.read_text(encoding="utf-8")
    assert captured["target_lang"] == "fr"
    assert "Language: fr" in out


def test_translate_po_infers_target_language_from_poedit_headers(monkeypatch, tmp_path):
    source = tmp_path / "messages.po"
    source.write_text(
        (
            'msgid ""\n'
            'msgstr ""\n'
            '"X-Poedit-Language: Portuguese\\n"\n'
            '"X-Poedit-Country: BRAZIL\\n"\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgid "Hello"\n'
            'msgstr ""\n'
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_translation(**kwargs):
        captured["target_lang"] = kwargs["target_lang"]
        kwargs["translator"].save(kwargs["store"], kwargs["output_path"])

    monkeypatch.setattr("aitran.translate._run_translation", fake_run_translation)

    translate_po(
        model=DEFAULT_TEST_MODEL,
        po_path=str(source),
        source_lang="en",
        target_lang="",
        verbose=False,
        output_path=str(source),
        context_file=None,
        context_length=4096,
    )

    out = source.read_text(encoding="utf-8")
    assert captured["target_lang"] == "pt_BR"
    assert "Language: pt_BR" in out
    assert "X-Poedit-Language: Portuguese" in out
    assert "X-Poedit-Country: BRAZIL" in out


def test_translate_po_without_lang_or_header_reports_error(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "messages.po"
    source.write_text(
        (
            'msgid ""\n'
            'msgstr ""\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgid "Hello"\n'
            'msgstr ""\n'
        ),
        encoding="utf-8",
    )

    def fake_run_translation(**_kwargs):
        pytest.fail("translation should not start without a target language")

    monkeypatch.setattr("aitran.translate._run_translation", fake_run_translation)

    translate_po(
        model=DEFAULT_TEST_MODEL,
        po_path=str(source),
        source_lang="en",
        target_lang="",
        verbose=False,
        output_path=str(source),
        context_file=None,
        context_length=4096,
    )

    captured = capsys.readouterr()
    assert "No target language specified via --lang or PO header" in captured.err


def test_translate_po_updates_last_translator_with_package_version(
    monkeypatch, tmp_path
):
    source = tmp_path / "messages.po"
    source.write_text(
        (
            'msgid ""\n'
            'msgstr ""\n'
            '"Last-Translator: Jane Doe <jane@example.com>\\n"\n'
            '"Language: zh_CN\\n"\n'
            '"Content-Type: text/plain; charset=UTF-8\\n"\n'
            "\n"
            'msgid "Hello"\n'
            'msgstr ""\n'
        ),
        encoding="utf-8",
    )

    def fake_run_translation(**kwargs):
        kwargs["translator"].save(kwargs["store"], kwargs["output_path"])

    monkeypatch.setattr("aitran.translate._run_translation", fake_run_translation)

    translate_po(
        model=DEFAULT_TEST_MODEL,
        po_path=str(source),
        source_lang="en",
        target_lang="",
        verbose=False,
        output_path=str(source),
        context_file=None,
        context_length=4096,
    )

    out = source.read_text(encoding="utf-8")
    assert f"Last-Translator: aitran v{package_version('aitran')}" in out
    assert "Jane Doe <jane@example.com>" not in out
    assert "aitran v0.1.0" not in out


# ── XliffTranslator apply_batch ───────────────────────────────────


def test_xliff_get_untranslated_respects_done_states_with_source_target_match():
    from translate.storage import xliff

    xlf = xliff.xlifffile.parsestring(
        rb"""<?xml version="1.0" encoding="UTF-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file source-language="en" target-language="zh-CN" datatype="plaintext">
    <body>
      <trans-unit id="final-same" approved="yes">
        <source>Aa</source>
        <target state="final">Aa</target>
      </trans-unit>
      <trans-unit id="translated-same">
        <source>Emoji</source>
        <target state="translated">Emoji</target>
      </trans-unit>
      <trans-unit id="needs-same">
        <source>Couldn't restore session</source>
        <target state="needs-translation">Couldn't restore session</target>
      </trans-unit>
      <trans-unit id="empty-target">
        <source>Translate me</source>
        <target/>
      </trans-unit>
      <trans-unit id="translate-no" translate="no">
        <source>App name</source>
        <target state="needs-translation">App name</target>
      </trans-unit>
    </body>
  </file>
</xliff>
"""
    )

    untranslated = XliffTranslator.get_untranslated(xlf)
    assert [unit.xmlelement.get("id") for unit in untranslated] == [
        "needs-same",
        "empty-target",
    ]


def test_xliff_apply_batch_fuzzy_state():
    from translate.storage import xliff

    xf = xliff.xlifffile()
    xu = xf.addsourceunit("hello")
    XliffTranslator.apply_batch(
        xf,
        [xu],
        [
            TranslatedUnit(index=1, targets=["你好"], fuzzy=True, note="ambiguous"),
        ],
    )
    out = bytes(xf).decode()
    assert 'state="needs-review-translation"' in out
    assert '<note from="translator">ambiguous</note>' in out


def test_xliff_apply_batch_clean_state():
    from translate.storage import xliff

    xf = xliff.xlifffile()
    xu = xf.addsourceunit("hello")
    XliffTranslator.apply_batch(
        xf,
        [xu],
        [
            TranslatedUnit(index=1, targets=["你好"], fuzzy=False),
        ],
    )
    out = bytes(xf).decode()
    assert 'state="translated"' in out


# ── build_model ────────────────────────────────────────────────────


def test_build_model_requires_colon():
    with pytest.raises(ValueError, match="provider:model"):
        build_model("gpt-4o-mini")


def test_build_model_anthropic_provider():
    m = build_model("anthropic:claude-sonnet-4-5", api_key="sk-test")
    assert isinstance(m, AnthropicModel)
    settings = m.settings
    assert settings["anthropic_cache_instructions"] is True
    assert settings["anthropic_cache"] == "5m"


def test_build_model_openai_provider():
    m = build_model(DEFAULT_TEST_MODEL, api_key="sk-test")
    assert isinstance(m, OpenAIChatModel)


# ── Agent instructions injection ────────────────────────────────────


def test_agent_instructions_inject_glossary():
    model = TestModel(
        custom_output_args={
            "translations": [{"index": 1, "targets": ["ok"], "fuzzy": False}],
        }
    )
    deps = TranslationDeps(
        source_lang="en",
        target_lang="zh_CN",
        context="A mobile banking app",
        dict_entries=[("login", "登录"), ("logout", "退出")],
        expected_indices=(1,),
    )
    agent = build_translator_agent(model)
    with agent.override(model=model):
        agent.run_sync(
            build_input_xml([FakeUnit("login")], start_index=1),
            deps=deps,
        )

    # Inspect the last model request to check instructions were injected
    last_req = model.last_model_request_parameters
    instruction_texts = [
        part.content if isinstance(part.content, str) else str(part.content)
        for part in last_req.instruction_parts or []
    ]
    combined = "\n".join(instruction_texts)
    assert "登录" in combined
    assert "A mobile banking app" in combined
    assert "en - English" in combined
    assert "zh_CN - Chinese (China)" in combined


def test_agent_instructions_reject_ambiguous_language_code():
    model = TestModel(
        custom_output_args={
            "translations": [{"index": 1, "targets": ["ok"], "fuzzy": False}],
        }
    )
    deps = TranslationDeps(
        source_lang="en",
        target_lang="zh",
        context="",
        dict_entries=[],
        expected_indices=(1,),
    )
    agent = build_translator_agent(model)
    with (
        agent.override(model=model),
        pytest.raises(ValueError, match="Unknown or ambiguous language code"),
    ):
        agent.run_sync(
            build_input_xml([FakeUnit("login")], start_index=1),
            deps=deps,
        )


# ── on_progress callback ───────────────────────────────────────────


async def test_translate_batch_on_progress_callback():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "targets": ["你好"], "fuzzy": False},
                {"index": 2, "targets": ["世界"], "fuzzy": False},
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("hello"), FakeUnit("world")]
    progress_items = []

    def track(src: str, result: TranslatedUnit):
        progress_items.append((src, result.targets[0], result.fuzzy))

    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps(),
            [],
            on_progress=track,
        )
    assert len(results) == 2
    assert len(progress_items) == 2
    assert progress_items[0] == ("hello", "你好", False)
    assert progress_items[1] == ("world", "世界", False)


# ── Output validator: extra indices ─────────────────────────────────


async def test_translate_batch_rejects_extra_indices():
    """Model returns an index not in the request — should trigger ModelRetry."""
    call_count = 0

    async def stream_fn(_messages, _agent_info):  # noqa: RUF029
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield (
                '{"translations":['
                '{"index":1,"targets":["ok"],"fuzzy":false},'
                '{"index":99,"targets":["bogus"],"fuzzy":false}'
                "]}"
            )
        else:
            yield ('{"translations":[{"index":1,"targets":["ok"],"fuzzy":false}]}')

    fn_model = FunctionModel(stream_function=stream_fn)
    agent = build_translator_agent(fn_model)
    units = [FakeUnit("a")]
    with agent.override(model=fn_model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert len(results) == 1
    assert call_count >= 2  # retried after extra index


# ── HTML entity unescaping ──────────────────────────────────────────


async def test_translate_batch_unescapes_html_entities():
    """format_as_xml escapes <>& in source; target should be unescaped back."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": ["点击 &lt;code&gt;btn&lt;/code&gt;"],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("click <code>btn</code>")]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert results[0].targets[0] == "点击 <code>btn</code>"


async def test_translate_batch_unescapes_mixed_markup_entities():
    """Toolkit entity decode reverses format_as_xml escaping for common tags."""
    encoded = (
        'Open &lt;a href="/docs?a=1&amp;b=2"&gt;docs&lt;/a&gt;, '
        "see &lt;strong&gt;bold&lt;/strong&gt;, "
        "&lt;code&gt;x &amp; y&lt;/code&gt; &lt;br/&gt;"
    )
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": [encoded],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [
        FakeUnit(
            'Open <a href="/docs?a=1&b=2">docs</a>, '
            "see <strong>bold</strong>, <code>x & y</code> <br/>"
        )
    ]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert results[0].targets[0] == (
        'Open <a href="/docs?a=1&amp;b=2">docs</a>, '
        "see <strong>bold</strong>, <code>x & y</code> <br/>"
    )


async def test_translate_batch_preserves_non_xml_text_entities():
    """Only XML text serialization entities should be decoded."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": ["版权 &copy; &amp; Co &quot;quoted&quot;"],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit('Copyright © & Co "quoted"')]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert results[0].targets[0] == "版权 &copy; & Co &quot;quoted&quot;"


async def test_translate_batch_unescapes_plain_ampersands():
    """Prompt XML escaping of plain ampersands should be reversed."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": ["AT&amp;T 和 Rock &amp; Roll"],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("AT&T and Rock & Roll")]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert results[0].targets[0] == "AT&T 和 Rock & Roll"


async def test_translate_batch_unescapes_numeric_placeholder_tags():
    """Numeric rich-text placeholders are XML-like tags, not HTML tags."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": ["&lt;0&gt;链接&lt;/0&gt;"],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("<0>link</0>")]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert results[0].targets[0] == "<0>链接</0>"


async def test_translate_batch_preserves_source_escaped_strings():
    """Entities already escaped in the source should stay escaped in target."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": ["显示 &lt;code&gt;btn&lt;/code&gt;"],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("show &lt;code&gt;btn&lt;/code&gt;")]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,)),
            [],
            on_progress=None,
        )
    assert results[0].targets[0] == "显示 &lt;code&gt;btn&lt;/code&gt;"


def test_translate_po_dir_runs_files_in_parallel(monkeypatch, tmp_path):
    """Directory PO translation should submit multiple files concurrently."""
    for name in ("a.po", "b.po", "c.po"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    active = 0
    max_active = 0
    calls: list[str] = []
    lock = threading.Lock()

    def fake_translate_po(*args, **_kwargs):
        nonlocal active, max_active
        po_path = args[1]
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(po_path)
        time.sleep(0.02)
        with lock:
            active -= 1

    monkeypatch.setattr("aitran.translate.translate_po", fake_translate_po)

    translate_po_dir(
        DEFAULT_TEST_MODEL,
        str(tmp_path),
        "en",
        "zh",
        False,
        None,
        4096,
        jobs=2,
    )

    assert len(calls) == 3
    assert max_active == 2


def test_translate_xliff_dir_runs_files_in_parallel(monkeypatch, tmp_path):
    """Directory XLIFF translation should submit multiple files concurrently."""
    for name in ("a.xlf", "b.xliff", "c.xlf"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "ignore.po").write_text("", encoding="utf-8")

    active = 0
    max_active = 0
    calls: list[str] = []
    lock = threading.Lock()

    def fake_translate_xliff_file(*args, **_kwargs):
        nonlocal active, max_active
        xliff_path = args[1]
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(xliff_path)
        time.sleep(0.02)
        with lock:
            active -= 1

    monkeypatch.setattr(
        "aitran.translate.translate_xliff_file", fake_translate_xliff_file
    )

    translate_xliff_dir(
        DEFAULT_TEST_MODEL,
        str(tmp_path),
        "en",
        "zh",
        False,
        None,
        4096,
        jobs=2,
    )

    assert len(calls) == 3
    assert max_active == 2


# ── Plural form support ────────────────────────────────────────────


@dataclass
class FakePluralUnit:
    source: multistring
    context: str | None = None
    _note: str | None = None

    def hasplural(self) -> bool:
        return len(self.source.strings) > 1

    def getcontext(self) -> str:
        return self.context or ""

    def getnotes(self) -> str:
        return self._note or ""


def test_build_input_xml_plural_units():
    """Plural units should include sources and plural_tags."""
    units = [
        FakePluralUnit(
            source=multistring([
                "{0} result",
                "{0} results",
            ])
        ),
    ]
    xml = build_input_xml(units, start_index=1, plural_tags=["one", "other"])
    assert "{0} result" in xml
    assert "{0} results" in xml
    assert "<sources>" in xml
    assert "plural_tags" not in xml


def test_build_input_xml_singular_no_plural_tags():
    """Singular units should not include plural_tags."""
    units = [FakeUnit("Hello")]
    xml = build_input_xml(units, start_index=1, plural_tags=["one", "other"])
    assert "<source>Hello</source>" in xml
    assert "plural_tags" not in xml
    assert "sources" not in xml


async def test_translate_batch_handles_plural_targets():
    """Agent returning targets list for plural units should work."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": ["{0} 个结果", "{0} 个结果"],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [
        FakePluralUnit(
            source=multistring(["{0} result", "{0} results"])
        ),
    ]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,), plural_tags=["one", "other"]),
            [],
            on_progress=None,
        )
    assert results[0].targets == ["{0} 个结果", "{0} 个结果"]


async def test_translate_batch_plural_targets_entity_decode():
    """XML entities in plural targets should be decoded."""
    model = TestModel(
        custom_output_args={
            "translations": [
                {
                    "index": 1,
                    "targets": [
                        "&lt;code&gt; 链接",
                        "&lt;code&gt; 链接",
                    ],
                    "fuzzy": False,
                },
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [
        FakePluralUnit(
            source=multistring([
                "<code> link",
                "<code> links",
            ])
        ),
    ]
    with agent.override(model=model):
        results = await _translate_batch(
            agent,
            units,
            1,
            _make_deps((1,), plural_tags=["one", "other"]),
            [],
            on_progress=None,
        )
    assert results[0].targets == ["<code> 链接", "<code> 链接"]
