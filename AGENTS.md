# Coding Agent Instructions

This file provides guidance to Coding agents when working with code in this repository.

## Build & Run

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_translate.py  # run a single test file
uv run pytest -k test_pattern          # run by keyword
uv run ruff check --fix                # lint + auto-fix
uv run ruff format                     # format

uvx prek install                       # install pre-commit hooks (uses `prek`, NOT `pre-commit`)
uv run aitran --po file.po -l zh       # run the CLI
```

Python 3.10+. Package manager: `uv`. Build backend: `uv_build`.

## Architecture

Single-package CLI at `src/aitran/`. Entry point: `aitran = "aitran.cli:app"` (Click group).

- `cli.py` — Click CLI with `translate` (default command), `sync`, `remove`, `userdict` subcommands
- `agent.py` — pydantic-ai agent definition, model routing (`build_model`), structured output types (`TranslatedUnit` / `TranslationBatch`), output validation
- `translate.py` — batch translation loop with streaming via `rich` progress bars; `PoTranslator` and `XliffTranslator` adapter classes handle format-specific parse/filter/apply/save
- `prompts/__init__.py` — inline system + user prompt strings (not external files)
- `dicts.py` — glossary lookup with cascading config discovery (CWD → git root → XDG dir)
- `manipulate.py` — PO entry removal by filter (fuzzy, obsolete, regex reference match)
- `sync.py` — update PO from POT preserving existing translations
- `utils.py` — config discovery, language code normalization, OS file-open helpers

### Data flow

1. CLI parses args → calls `translate_po` / `translate_xliff_file`
2. Adapter parses file, filters units needing translation
3. `_run_translation_async` batches units by accumulated char length (`--context-length`, default 4096)
4. Each batch: `build_input_xml()` serializes units → agent streams results → `html.unescape()` reverses XML escaping → adapter applies results and saves
5. Progress rendered via `rich.progress.Progress` with per-unit verbose output

### Model routing

`build_model()` in `agent.py` splits on `:` to get provider:model. Anthropic gets special `AnthropicModel` with prompt caching; all other providers route through `OpenAIChatModel` using pydantic-ai's `infer_provider_class()`. Unknown providers fall back to `OpenAIProvider` (OpenAI-compatible gateways).

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
- **Output validation**: Agent validates index completeness via `@agent.output_validator` — missing/extra indices trigger `ModelRetry` (up to 3 retries).
- **HTML escaping**: `format_as_xml` escapes `<>&` in source; `_translate_batch` calls `html.unescape()` on targets to reverse this.
- **Rate limiting**: HTTP 429 triggers a 20-second sleep before retry. Timeouts (408/504) retry immediately.
