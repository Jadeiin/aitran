"""Tests for the Pydantic AI translator agent and file I/O adapters."""

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
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
)

# ── Helpers ──────────────────────────────────────────────────────


class FakeUnit:
    def __init__(
        self, source: str, context: str | None = None, comment: str | None = None
    ):
        self.source = source
        self.context = context
        self.comment = comment


def _make_deps(expected_indices=(1, 2)):
    return TranslationDeps(
        source_lang="en",
        target_lang="zh",
        context="",
        dict_entries=[],
        expected_indices=expected_indices,
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
    units = [FakeUnit("File", context="Menu", comment="top-level")]
    xml = build_input_xml(units, start_index=5)
    assert "<index>5</index>" in xml
    assert "<context>Menu</context>" in xml
    assert "<comment>top-level</comment>" in xml


def test_build_input_xml_omits_none_fields():
    units = [FakeUnit("Plain")]
    xml = build_input_xml(units, start_index=1)
    assert "context" not in xml
    assert "comment" not in xml
    assert "null" not in xml


# ── Translate batch via TestModel ─────────────────────────────────


async def test_translate_batch_success():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "target": "你好", "fuzzy": False, "note": None},
                {"index": 2, "target": "世界", "fuzzy": False, "note": None},
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
    assert results[0].target == "你好"
    assert results[1].target == "世界"


async def test_translate_batch_fuzzy_flag():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "target": "x", "fuzzy": True, "note": "unsure"},
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

    async def stream_fn(messages, agent_info):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield '{"translations":[{"index":1,"target":"ok","fuzzy":false}]}'
        else:
            yield '{"translations":[{"index":1,"target":"ok","fuzzy":false},{"index":2,"target":"yes","fuzzy":false}]}'

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
                {"index": 1, "target": "你好", "fuzzy": False, "note": None},
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
                    {"index": 2, "target": "世界", "fuzzy": False, "note": None},
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
            TranslatedUnit(index=1, target="你好", fuzzy=True, note="check me"),
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
            TranslatedUnit(index=1, target="你好", fuzzy=False),
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
            TranslatedUnit(index=1, target="你好", fuzzy=False),
        ],
    )
    out = bytes(pf).decode()
    assert 'msgstr "你好"' in out


# ── XliffTranslator apply_batch ───────────────────────────────────


def test_xliff_apply_batch_fuzzy_state():
    from translate.storage import xliff

    xf = xliff.xlifffile()
    xu = xf.addsourceunit("hello")
    XliffTranslator.apply_batch(
        xf,
        [xu],
        [
            TranslatedUnit(index=1, target="你好", fuzzy=True, note="ambiguous"),
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
            TranslatedUnit(index=1, target="你好", fuzzy=False),
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
    m = build_model("openai:gpt-4o-mini", api_key="sk-test")
    assert isinstance(m, OpenAIChatModel)


# ── Agent instructions injection ────────────────────────────────────


def test_agent_instructions_inject_glossary():
    model = TestModel(
        custom_output_args={
            "translations": [{"index": 1, "target": "ok", "fuzzy": False}],
        }
    )
    deps = TranslationDeps(
        source_lang="en",
        target_lang="zh",
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
    assert "en" in combined and "zh" in combined


# ── on_progress callback ───────────────────────────────────────────


async def test_translate_batch_on_progress_callback():
    model = TestModel(
        custom_output_args={
            "translations": [
                {"index": 1, "target": "你好", "fuzzy": False},
                {"index": 2, "target": "世界", "fuzzy": False},
            ],
        }
    )
    agent = build_translator_agent(model)
    units = [FakeUnit("hello"), FakeUnit("world")]
    progress_items = []

    def track(src: str, result: TranslatedUnit):
        progress_items.append((src, result.target, result.fuzzy))

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

    async def stream_fn(messages, agent_info):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield '{"translations":[{"index":1,"target":"ok","fuzzy":false},{"index":99,"target":"bogus","fuzzy":false}]}'
        else:
            yield '{"translations":[{"index":1,"target":"ok","fuzzy":false}]}'

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
                    "target": "点击 &lt;code&gt;btn&lt;/code&gt;",
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
    assert results[0].target == "点击 <code>btn</code>"
