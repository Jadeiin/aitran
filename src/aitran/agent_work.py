"""Run-local document work exposed to translator and reviewer agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import FunctionToolset, ModelRetry, RunContext, ToolDefinition

from aitran.agents._base import prompt_texts, safe_prompt_text
from aitran.dicts import find_matching_entries
from aitran.qa import QARunner, UnitQAReport
from aitran.toolsets._base import OrchestratorDeps


class TranslationTask(BaseModel):
    """Canonical task consumed directly from a translator run prompt."""

    model_config = ConfigDict(extra="ignore")

    input_path: str = Field(
        validation_alias=AliasChoices("input_path", "input_file", "path")
    )
    output_path: str = Field(
        validation_alias=AliasChoices("output_path", "output_file")
    )
    source_language: str = Field(
        default="en", validation_alias=AliasChoices("source_language", "source_lang")
    )
    target_language: str = Field(
        default="", validation_alias=AliasChoices("target_language", "target_lang")
    )
    scope: Literal["pending", "repair", "all"] = "pending"
    context_file: str | None = None
    max_page_units: int = Field(default=100, gt=0)
    max_page_chars: int = Field(default=4096, gt=0)
    order: Literal["file", "source", "reference", "context"] = "file"
    profile: Literal["fast", "full"] = Field(
        default="full",
        description="Unit metadata detail level; never free-form guidance.",
    )
    instructions: str = Field(
        default="",
        description="Free-form domain, tone, terminology, or product guidance.",
    )

    @model_validator(mode="before")
    @classmethod
    def relocate_prose_profile(cls, value: Any) -> Any:
        """Preserve misplaced LLM guidance without accepting a bogus profile.

        Returns:
            Normalized task input.
        """
        if not isinstance(value, dict):
            return value
        profile = value.get("profile")
        if not isinstance(profile, str) or profile in {"fast", "full"}:
            return value
        guidance = profile.strip()
        if not guidance:
            return value
        normalized = dict(value)
        existing = str(normalized.get("instructions") or "").strip()
        normalized["instructions"] = f"{existing}\n{guidance}" if existing else guidance
        normalized["profile"] = "full"
        return normalized


class ReviewTask(BaseModel):
    """Canonical task consumed directly from a reviewer run prompt."""

    model_config = ConfigDict(extra="ignore")

    input_path: str = Field(
        validation_alias=AliasChoices("input_path", "input_file", "path")
    )
    output_path: str = Field(
        validation_alias=AliasChoices("output_path", "output_file")
    )
    source_language: str = Field(
        default="en", validation_alias=AliasChoices("source_language", "source_lang")
    )
    target_language: str = Field(
        default="", validation_alias=AliasChoices("target_language", "target_lang")
    )
    max_page_units: int = Field(default=100, gt=0)
    max_page_chars: int = Field(default=4096, gt=0)
    instructions: str = Field(
        default="",
        description="Free-form domain, tone, terminology, or product guidance.",
    )


class TranslationResult(BaseModel):
    """One translation submitted for a retrieved unit."""

    index: int
    targets: list[str] = Field(min_length=1)
    fuzzy: bool = False
    note: str | None = None


class ReviewResult(BaseModel):
    """One sparse problem report submitted for a reviewed unit."""

    index: int
    verdict: Literal["revise", "reject"]
    corrected: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def revise_has_correction(self) -> ReviewResult:
        """Require an actionable correction for a revise verdict.

        Returns:
            Validated result.

        Raises:
            ValueError: If a revise verdict has no correction.
        """
        if self.verdict == "revise" and self.corrected is None:
            raise ValueError("revise results require corrected text")
        return self


class UnitView(BaseModel):
    """Format-neutral view of one PO/XLIFF translation unit."""

    index: int
    sources: list[str]
    targets: list[str] | None = None
    context: str | None = None
    location: list[str] = Field(default_factory=list)
    note: str | None = None
    flags: list[str] = Field(default_factory=list)
    qa_errors: list[str] = Field(default_factory=list)


class TranslationPage(BaseModel):
    """One translator retrieval page."""

    batch_id: str
    total: int
    source_language: str
    target_language: str
    plural_tags: list[str] | None
    context: str
    glossary: list[dict[str, str]]
    units: list[UnitView]


class ReviewPage(BaseModel):
    """One reviewer retrieval page."""

    batch_id: str
    total: int
    source_language: str
    target_language: str
    units: list[UnitView]


class BatchProgress(BaseModel):
    """Progress returned after a page is safely written."""

    accepted: int
    total: int
    remaining: int
    complete: bool


class ArtifactResult(BaseModel):
    """Exact completed file artifact returned to the caller."""

    status: Literal["translated", "reviewed"]
    input_path: str
    output_path: str
    processed_units: int
    summary: dict[str, int] | None = None


@dataclass
class _Page:
    batch_id: str
    items: list[tuple[int, Any]]


@dataclass
class _TranslationState:
    task: TranslationTask
    adapter: Any
    store: Any
    items: list[tuple[int, Any]]
    source_language: str
    target_language: str
    context: str
    plural_tags: list[str] | None
    cursor: int = 0
    current: _Page | None = None
    last_batch_id: str | None = None
    last_progress: BatchProgress | None = None
    stage: Literal["get", "submit", "finish"] = "get"


@dataclass
class _ReviewState:
    task: ReviewTask
    adapter: Any
    store: Any
    items: list[tuple[int, Any, UnitQAReport]]
    all_units: dict[int, Any]
    source_language: str
    target_language: str
    summary: dict[str, int] = field(
        default_factory=lambda: {"pass": 0, "revise": 0, "reject": 0}
    )
    cursor: int = 0
    current: _Page | None = None
    last_batch_id: str | None = None
    last_progress: BatchProgress | None = None
    stage: Literal["get", "submit", "finish"] = "get"


def _prompt_json(ctx: RunContext[OrchestratorDeps], model: type[BaseModel]) -> Any:
    prompt = ctx.prompt
    if not isinstance(prompt, str):
        raise TypeError("The delegated task must be a JSON string")
    return model.model_validate_json(prompt)


def _adapter(path: str) -> Any:
    from aitran.translate import PoTranslator, XliffTranslator

    if path.lower().endswith((".po", ".pot")):
        return PoTranslator()
    if path.lower().endswith((".xliff", ".xlf")):
        return XliffTranslator()
    raise ValueError(f"Unsupported translation file: {path}")


def _all_translatable(adapter: Any, store: Any) -> list[Any]:
    from aitran.translate import PoTranslator

    if isinstance(adapter, PoTranslator):
        return [
            unit
            for unit in store.units
            if not unit.isheader() and not getattr(unit, "isobsolete", lambda: False)()
        ]
    return [unit for unit in store.units if adapter._get_translate_flag(unit)]


def _translated_units(adapter: Any, store: Any) -> list[Any]:
    from aitran.translate import PoTranslator

    if isinstance(adapter, PoTranslator):
        return [
            unit
            for unit in store.units
            if unit.source and not unit.isheader() and unit.target
        ]
    return [
        unit
        for unit in store.units
        if (unit.source or "").strip() and (unit.target or "").strip()
    ]


def _languages(adapter: Any, store: Any, source: str, target: str) -> tuple[str, str]:
    from aitran.translate import PoTranslator

    if isinstance(adapter, PoTranslator):
        resolved_target = target or adapter.get_target_language(store) or ""
        resolved_source = source or "en"
    else:
        resolved_target = target or store.targetlanguage or ""
        resolved_source = source or store.sourcelanguage or "en"
    if not resolved_target:
        raise ValueError("Target language is required and could not be inferred")
    return resolved_source, resolved_target


def _unit_chars(unit: Any) -> int:
    values = [*prompt_texts(unit.source), *prompt_texts(unit.target or "")]
    return sum(len(value) for value in values)


def _take_page(items: list[Any], cursor: int, max_units: int, max_chars: int) -> int:
    stop = cursor
    chars = 0
    while stop < len(items) and stop - cursor < max_units:
        item = items[stop]
        unit = item[1]
        size = _unit_chars(unit)
        if stop > cursor and chars + size > max_chars:
            break
        chars += size
        stop += 1
    return stop


def _view(
    index: int,
    unit: Any,
    *,
    include_target: bool,
    include_metadata: bool = True,
    report: UnitQAReport | None = None,
) -> UnitView:
    get_context = getattr(unit, "getcontext", None)
    get_locations = getattr(unit, "getlocations", None)
    get_notes = getattr(unit, "getnotes", None)
    context = get_context() if callable(get_context) else None
    locations = list(get_locations() or []) if callable(get_locations) else []
    note = get_notes().strip() if callable(get_notes) else None
    raw_flags = getattr(unit, "typecomments", None) or []
    if isinstance(raw_flags, str):
        raw_flags = [flag.strip() for flag in raw_flags.split(",") if flag.strip()]
    errors = []
    if report is not None:
        errors = [
            f"[{error.severity}] {error.checker}: {error.message}"
            for error in report.errors
        ]
    return UnitView(
        index=index,
        sources=prompt_texts(unit.source),
        targets=prompt_texts(unit.target) if include_target else None,
        context=safe_prompt_text(context) if context and include_metadata else None,
        location=(
            [safe_prompt_text(location) for location in locations]
            if include_metadata
            else []
        ),
        note=safe_prompt_text(note) if note and include_metadata else None,
        flags=(
            [safe_prompt_text(flag) for flag in raw_flags] if include_metadata else []
        ),
        qa_errors=errors,
    )


def _visible(state: Any, stage: str):
    def prepare(
        _ctx: RunContext[OrchestratorDeps], tool: ToolDefinition
    ) -> ToolDefinition | None:
        return tool if state.stage == stage else None

    return prepare


def _translation_state(task: TranslationTask) -> _TranslationState:
    from aitran.translate import (
        PoTranslator,
        _last_translator,
        _order_units,
        _read_context,
    )

    adapter = _adapter(task.input_path)
    store = adapter.parse(task.input_path)
    source, target = _languages(
        adapter, store, task.source_language, task.target_language
    )
    if isinstance(adapter, PoTranslator):
        store.updateheader(
            add=True, **{"Last-Translator": _last_translator(), "Language": target}
        )
    else:
        for unit in store.units:
            if not getattr(unit, "_source_locale_set", False):
                unit.xmlelement.set("source-language", source)
                unit.xmlelement.set("target-language", target)
                unit._source_locale_set = True

    all_units = _all_translatable(adapter, store)
    stable = {id(unit): index for index, unit in enumerate(all_units, start=1)}
    if task.scope == "pending":
        selected = adapter.get_untranslated(store)
    elif task.scope == "repair":
        pending = {id(unit) for unit in adapter.get_untranslated(store)}
        reports = QARunner(target_lang=target).check_units(all_units)
        selected = [
            unit
            for unit, report in zip(all_units, reports, strict=True)
            if id(unit) in pending or report.has_errors
        ]
    else:
        selected = all_units
    selected = _order_units(selected, task.order)
    items = [(stable[id(unit)], unit) for unit in selected]
    plural_tags = store.get_plural_tags() if hasattr(store, "get_plural_tags") else None
    state = _TranslationState(
        task=task,
        adapter=adapter,
        store=store,
        items=items,
        source_language=source,
        target_language=target,
        context=_read_context(task.context_file),
        plural_tags=plural_tags,
    )
    if not items:
        state.stage = "finish"
    return state


def translation_toolset(
    ctx: RunContext[OrchestratorDeps],
) -> FunctionToolset[OrchestratorDeps]:
    """Create the translator's isolated toolset for one Agent run.

    Returns:
        Fresh run-local toolset.
    """
    task = _prompt_json(ctx, TranslationTask)
    state = _translation_state(task)
    tools = FunctionToolset[OrchestratorDeps](
        sequential=True, include_return_schema=True, id="translation-work"
    )

    @tools.tool(prepare=_visible(state, "get"))
    def get_translation_batch(
        _ctx: RunContext[OrchestratorDeps],
    ) -> TranslationPage:
        """Retrieve the next bounded page of translation units.

        Returns:
            Current translation page.
        """
        stop = _take_page(
            state.items,
            state.cursor,
            task.max_page_units,
            task.max_page_chars,
        )
        current_items = state.items[state.cursor : stop]
        state.current = _Page(uuid.uuid4().hex, current_items)
        state.stage = "submit"
        units = [unit for _, unit in current_items]
        glossary = [
            {"source": source, "target": target}
            for source, target in find_matching_entries(
                [unit.source for unit in units], state.target_language
            )
        ]
        return TranslationPage(
            batch_id=state.current.batch_id,
            total=len(state.items),
            source_language=state.source_language,
            target_language=state.target_language,
            plural_tags=state.plural_tags,
            context=state.context,
            glossary=glossary,
            units=[
                _view(
                    index,
                    unit,
                    include_target=True,
                    include_metadata=task.profile == "full",
                )
                for index, unit in current_items
            ],
        )

    @tools.tool(prepare=_visible(state, "submit"))
    def submit_translation_batch(
        _ctx: RunContext[OrchestratorDeps],
        batch_id: str,
        translations: list[TranslationResult],
    ) -> BatchProgress:
        """Validate, apply, and save every translation in the current page.

        Returns:
            Updated progress.

        Raises:
            ModelRetry: If the batch identity or submitted units are invalid.
        """
        if batch_id == state.last_batch_id and state.last_progress is not None:
            return state.last_progress
        current = state.current
        if current is None or batch_id != current.batch_id:
            raise ModelRetry("The batch_id is stale or does not match the current page")
        expected = {index for index, _ in current.items}
        by_index = {result.index: result for result in translations}
        if len(by_index) != len(translations) or set(by_index) != expected:
            raise ModelRetry(
                "Return every requested index exactly once; "
                f"expected {sorted(expected)}, "
                f"got {sorted(by_index)}"
            )
        from aitran.translate import _decode_serialized_markup, _multistring

        ordered = [by_index[index] for index, _ in current.items]
        units = [unit for _, unit in current.items]
        for unit, result in zip(units, ordered, strict=True):
            sources = _multistring(unit.source)
            expected_forms = len(state.plural_tags or []) if unit.hasplural() else 1
            if expected_forms and len(result.targets) != expected_forms:
                raise ModelRetry(
                    f"Index {result.index} requires {expected_forms} target form(s)"
                )
            if len(result.targets) == 1 and len(sources) > 1:
                decode_sources = ["\n".join(sources)]
            else:
                decode_sources = sources
            result.targets = [
                _decode_serialized_markup(
                    str(decode_sources[min(i, len(decode_sources) - 1)]), target
                )
                for i, target in enumerate(result.targets)
            ]
        state.adapter.apply_batch(state.store, units, ordered)
        state.adapter.save(state.store, task.output_path)
        state.cursor += len(current.items)
        progress = BatchProgress(
            accepted=state.cursor,
            total=len(state.items),
            remaining=len(state.items) - state.cursor,
            complete=state.cursor == len(state.items),
        )
        state.last_batch_id = batch_id
        state.last_progress = progress
        state.current = None
        state.stage = "finish" if progress.complete else "get"
        return progress

    @tools.tool(prepare=_visible(state, "finish"))
    def finish_translation(_ctx: RunContext[OrchestratorDeps]) -> str:
        """Finalize the completed artifact and return its exact path as JSON.

        Returns:
            Serialized artifact result.
        """
        state.adapter.save(state.store, task.output_path)
        return ArtifactResult(
            status="translated",
            input_path=task.input_path,
            output_path=task.output_path,
            processed_units=len(state.items),
        ).model_dump_json()

    return tools


def _review_state(task: ReviewTask) -> _ReviewState:
    adapter = _adapter(task.input_path)
    store = adapter.parse(task.input_path)
    source, target = _languages(
        adapter, store, task.source_language, task.target_language
    )
    units = _translated_units(adapter, store)
    indexed = list(enumerate(units, start=1))
    reports = QARunner(target_lang=target).check_units(units)
    items = [
        (index, unit, report)
        for (index, unit), report in zip(indexed, reports, strict=True)
    ]
    state = _ReviewState(
        task=task,
        adapter=adapter,
        store=store,
        items=items,
        all_units=dict(indexed),
        source_language=source,
        target_language=target,
    )
    if not items:
        state.stage = "finish"
    return state


def review_toolset(
    ctx: RunContext[OrchestratorDeps],
) -> FunctionToolset[OrchestratorDeps]:
    """Create the reviewer's isolated toolset for one Agent run.

    Returns:
        Fresh run-local toolset.
    """
    task = _prompt_json(ctx, ReviewTask)
    state = _review_state(task)
    tools = FunctionToolset[OrchestratorDeps](
        sequential=True, include_return_schema=True, id="review-work"
    )

    @tools.tool(prepare=_visible(state, "get"))
    def get_review_batch(_ctx: RunContext[OrchestratorDeps]) -> ReviewPage:
        """Retrieve the next bounded page of translations and QA findings.

        Returns:
            Current review page.
        """
        stop = _take_page(
            state.items,
            state.cursor,
            task.max_page_units,
            task.max_page_chars,
        )
        page_items = state.items[state.cursor : stop]
        state.current = _Page(
            uuid.uuid4().hex, [(index, unit) for index, unit, _ in page_items]
        )
        state.stage = "submit"
        return ReviewPage(
            batch_id=state.current.batch_id,
            total=len(state.items),
            source_language=state.source_language,
            target_language=state.target_language,
            units=[
                _view(index, unit, include_target=True, report=report)
                for index, unit, report in page_items
            ],
        )

    @tools.tool(prepare=_visible(state, "submit"))
    def submit_review_batch(
        _ctx: RunContext[OrchestratorDeps],
        batch_id: str,
        reviews: list[ReviewResult],
    ) -> BatchProgress:
        """Apply sparse review findings and save the current page.

        Returns:
            Updated progress.

        Raises:
            ModelRetry: If the batch identity or submitted units are invalid.
        """
        if batch_id == state.last_batch_id and state.last_progress is not None:
            return state.last_progress
        current = state.current
        if current is None or batch_id != current.batch_id:
            raise ModelRetry("The batch_id is stale or does not match the current page")
        valid = {index for index, _ in current.items}
        indices = [review.index for review in reviews]
        if len(indices) != len(set(indices)) or not set(indices) <= valid:
            raise ModelRetry(
                f"Review indices must be unique and within {sorted(valid)}; "
                f"got {indices}"
            )
        state.adapter.apply_review_batch(
            state.store, state.all_units, reviews, auto_fix=True
        )
        for review in reviews:
            state.summary[review.verdict] += 1
        state.summary["pass"] += len(valid) - len(reviews)
        state.adapter.save(state.store, task.output_path)
        state.cursor += len(current.items)
        progress = BatchProgress(
            accepted=state.cursor,
            total=len(state.items),
            remaining=len(state.items) - state.cursor,
            complete=state.cursor == len(state.items),
        )
        state.last_batch_id = batch_id
        state.last_progress = progress
        state.current = None
        state.stage = "finish" if progress.complete else "get"
        return progress

    @tools.tool(prepare=_visible(state, "finish"))
    def finish_review(_ctx: RunContext[OrchestratorDeps]) -> str:
        """Finalize the reviewed artifact and return its exact path as JSON.

        Returns:
            Serialized artifact result.
        """
        state.adapter.save(state.store, task.output_path)
        return ArtifactResult(
            status="reviewed",
            input_path=task.input_path,
            output_path=task.output_path,
            processed_units=len(state.items),
            summary=state.summary,
        ).model_dump_json()

    return tools
