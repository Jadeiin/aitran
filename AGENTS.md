# Coding Agent Instructions

This file provides guidance to Coding agents when working with code in this repository.

## Build & Run

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_translate.py  # run a single test file
uv run pytest tests/test_app.py        # run interactive app tests
uv run pytest -k test_pattern          # run by keyword
uv run ruff check --fix                # lint + auto-fix
uv run ruff format                     # format

uvx prek install                       # install pre-commit hooks (uses `prek`, NOT `pre-commit`)
uv run aitran --po file.po -l zh       # run the CLI
uv run aitran                          # launch the interactive app
uv run aitran --prompt "translate this project to zh"  # one-shot app prompt
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

- `cli.py` — Click CLI; bare `aitran` launches the interactive app, and `translate`, `review`, `sync`, `remove`, `userdict`, `crowdin`, `weblate` remain as subcommands
- `app.py` — interactive app session management, REPL, resume handling, and deferred-tool approval loop
- `agents/` — pydantic-ai agent definitions:
  - `_base.py` — model routing (`build_model`), XML prompt builder (`build_input_xml`), shared helpers
  - `translator.py` — translator agent (`build_translator_agent`), output types (`TranslatedUnit` / `TranslationBatch`), prompts
  - `reviewer.py` — reviewer agent (`build_reviewer_agent`), output types (`ReviewedUnit` / `ReviewBatch`)
- `translate.py` — batch translation loop with streaming via `rich` progress bars; `PoTranslator` and `XliffTranslator` adapter classes handle format-specific parse/filter/apply/save
- `dicts.py` — glossary lookup with cascading config discovery (CWD → git root → XDG dir)
- `manipulate.py` — PO entry removal by filter (fuzzy, obsolete, regex reference match)
- `sync.py` — update PO from POT preserving existing translations
- `utils.py` — config discovery, language code normalization, OS file-open helpers
- `crowdin.py` — Crowdin API client: list projects/files/languages, progress, download/upload translations (XLIFF only)
- `weblate.py` — Weblate API client (`wlc`): list objects, stats, download/upload translations (PO/XLIFF)

### Data flow

1. CLI parses args → calls `translate_po` / `translate_xliff_file`
2. Adapter parses file, filters units needing translation
3. `_run_translation_async` batches units by accumulated char length (`--context-length`, default 4096)
4. Each batch: `build_input_xml()` serializes units → agent streams results → translate-toolkit entity decoding reverses XML escaping → adapter applies results through format APIs and saves
5. Progress rendered via `rich.progress.Progress` with per-unit verbose output

### Model routing

`build_model()` in `agents/_base.py` splits on `:` to get provider:model. Anthropic gets special `AnthropicModel` with prompt caching; all other providers route through `OpenAIChatModel` using pydantic-ai's `infer_provider_class()`. Unknown providers fall back to `OpenAIProvider` (OpenAI-compatible gateways).

### Interactive app

- Top-level entry: `aitran`
- One-shot prompt: `aitran --prompt "..."`
- Resume: `aitran --resume --session-id <id>`
- Environment for the app entry uses `AITRAN_APP_*` orchestrator settings (`AITRAN_APP_MODEL`, `AITRAN_APP_KEY`, `AITRAN_APP_HOST`, `AITRAN_APP_TMP`, `AITRAN_APP_AUTO_APPROVE`)

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
- **Plural handling**: `TranslatedUnit.targets` is always a list — length 1 for singular, length matching `plural_tags` for plural units. PO plural sources are passed via `sources` list in prompt XML; `plural_tags` (e.g. `["one", "other"]`) is injected via task instructions.
- **HTML/XML escaping**: `format_as_xml` escapes `<>&` in source text when building the prompt XML. `_decode_serialized_markup()` in `translate.py` conditionally reverses this by decoding only entities (`&amp;`, `&lt;`, `&gt;`) whose corresponding characters appeared in the original source. The prompt also explicitly instructs the LLM not to escape output. Both layers exist because LLMs are unreliable at following XML-escaping instructions.
- **XLIFF mutation**: Do not edit XLIFF XML nodes manually when applying translations. Use `xliffunit.settarget()`, `marktranslated()`, `markreviewneeded()`, and note APIs so translate-toolkit owns node creation, XML-safe text, and state mapping.
- **Rate limiting**: HTTP 429 triggers a 20-second sleep before retry. Timeouts (408/504) retry immediately.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
