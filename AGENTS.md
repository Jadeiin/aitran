# Coding Agent Instructions

This file provides guidance to Coding agents when working with code in this repository.

## Build & Run

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_translate.py  # run a single test file
uv run pytest tests/test_app.py        # run WebUI wiring tests
uv run pytest -k test_pattern          # run by keyword
uv run ruff check --fix                # lint + auto-fix
uv run ruff format                     # format

uvx prek install                       # install pre-commit hooks (uses `prek`, NOT `pre-commit`)
uv run aitran translate --po file.po -l zh  # run headless translation
uv run aitran                          # launch the orchestrator WebUI
uv run aitran --web-host 0.0.0.0 --web-port 9000
```

Python 3.10+. Package manager: `uv`. Build backend: `uv_build`.

## Format Scope

`aitran` intentionally focuses on the two mainstream bilingual translation
formats used by the core workflow: Gettext PO/POT and XLIFF/XLF. Do not add
native translation paths for arbitrary source formats unless the product scope
changes. Other file types should be translated by converting them to PO or XLIFF
first, using translate-toolkit's bundled converters or another reliable
round-trip pipeline, then converting the translated PO/XLIFF back to the source
format.

## Architecture

Single-package CLI at `src/aitran/`. Entry point: `aitran = "aitran.cli:app"` (Click group).

- `cli.py` — Click CLI; bare `aitran` launches the orchestrator WebUI, while `translate`, `review`, `sync`, `remove`, `userdict`, `crowdin`, `weblate` remain headless subcommands
- `app.py` — thin `Agent.to_web()` + Uvicorn adapter for the orchestrator WebUI
- `agents/` — pydantic-ai agent definitions:
  - `_base.py` — model routing (`build_model`) and shared prompt helpers
  - `translator.py` — complete-file translator Agent delegated directly through Pydantic AI Harness
  - `reviewer.py` — complete-file reviewer Agent delegated directly through Pydantic AI Harness
- `agent_work.py` — run-local paginated retrieval/submission toolsets and artifact contracts for translator/reviewer Agents
- `headless.py` — thin direct-Agent adapter for the headless translate/review commands
- `translate.py` — `PoTranslator` and `XliffTranslator` document adapters for format-specific parse/filter/apply/save
- `dicts.py` — glossary lookup with cascading config discovery (CWD → git root → XDG dir)
- `manipulate.py` — PO entry removal by filter (fuzzy, obsolete, regex reference match)
- `sync.py` — update PO from POT preserving existing translations
- `utils.py` — config discovery, language code normalization, OS file-open helpers
- `crowdin.py` — Crowdin API client: list projects/files/languages, progress, download/upload translations (XLIFF only)
- `weblate.py` — Weblate API client (`wlc`): list objects, stats, download/upload translations (PO/XLIFF)

### Data flow

1. WebUI orchestrator delegates one canonical JSON file task directly to the translator or reviewer Agent; headless commands run the same Agents directly.
2. A fresh run-local toolset parses the document, selects stable unit IDs, and exposes one bounded page at a time.
3. The Agent retrieves and submits each page; the toolset validates IDs/plurals, applies results through the format adapter, and saves after every accepted page.
4. The initial task remains in Agent history while pages contain only current batch data; `finish_translation` / `finish_review` returns the exact output artifact JSON.
5. Headless commands consume native `FunctionToolResultEvent` events and update one Rich stderr progress row per file from each accepted page's `BatchProgress`.

Delegated task JSON uses `instructions` for free-form domain, tone, terminology,
or product guidance. `profile` is strictly `fast` or `full` and controls only
retrieved unit metadata detail; never put prose in `profile`, `scope`, or `order`.

### Model routing

`build_model()` in `agents/_base.py` splits on `:` to get provider:model. Anthropic gets special `AnthropicModel` with prompt caching; all other providers route through `OpenAIChatModel` using pydantic-ai's `infer_provider_class()`. Unknown providers fall back to `OpenAIProvider` (OpenAI-compatible gateways).

### Web app

- Top-level entry: `aitran` serves Pydantic AI's WebUI
- Bind options: `--web-host`, `--web-port`
- Environment uses `AITRAN_APP_*` model settings plus `AITRAN_WEB_HOST` / `AITRAN_WEB_PORT`
- Write tools and Harness delegation use native deferred approval; do not reintroduce a separate toolguard layer
- Harness `SubAgents` must use `agent_folders=None` and `inherit_tools=False` so only the explicit translator/reviewer Agents are loaded and platform tools are not inherited

## Conventions & Gotchas

- **Model format**: `provider:model` with a colon. `build_model()` raises `ValueError` otherwise.
- **Async tests**: `asyncio_mode = "auto"` in pytest config — no `@pytest.mark.asyncio` decorator needed.
- **Mock models**: Use pydantic-ai's `TestModel` for sync outputs, `FunctionModel` with `stream_function` for testing retries.
- **Ruff**: `preview = true`, google docstring convention, `D`/`DOC` rules disabled in `tests/**`.
- **Pre-commit**: Managed by `prek`, NOT standard `pre-commit`. Use `uvx prek install`.
- **China PyPI mirror**: `pyproject.toml` configures `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` as default uv index. Remove this line if you're outside China and installs fail.
- **API host gets `/v1` appended**: In `translate.py:304`, a custom `--host` gets `rstrip("/") + "/v1"` appended automatically.
- **User dictionaries**: Looked up in order: `$CWD/.aitran/` → git root `.aitran/` → XDG user config dir (`platformdirs`). Named `dictionary-<lang>.json`.
- **Commitizen**: Conventional commits with `tag_format = "v$version"`, `major_version_zero = true`.
- **Output validation**: submit tools validate batch identity and index completeness; invalid submissions raise `ModelRetry`. Agent output validators require the corresponding finish tool artifact.
- **Plural handling**: `TranslatedUnit.targets` is always a list — length 1 for singular, length matching `plural_tags` for plural units. PO plural sources are passed via `sources` list in prompt XML; `plural_tags` (e.g. `["one", "other"]`) is injected via task instructions.
- **HTML/XML escaping**: Pydantic AI serializes tool payloads safely. `_decode_serialized_markup()` in `translate.py` conditionally reverses model-returned entities (`&amp;`, `&lt;`, `&gt;`) only when the corresponding character appeared in the original source.
- **XLIFF mutation**: Do not edit XLIFF XML nodes manually when applying translations. Use `xliffunit.settarget()`, `marktranslated()`, `markreviewneeded()`, and note APIs so translate-toolkit owns node creation, XML-safe text, and state mapping.
- **Rate limiting**: HTTP 429 triggers a 20-second sleep before retry. Timeouts (408/504) retry immediately.
