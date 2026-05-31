# aitran

Agentic translation workflows for localization platforms, plus direct
PO/XLIFF translation — streaming progress, deferred approval, and
translator notes for human review.
Successor to [gpt-po](https://github.com/ryanhex53/gpt-po).

Supports OpenAI, Anthropic, DeepSeek, and any OpenAI-compatible provider.

## Installation

```bash
# Recommended: uv
uv tool install git+https://github.com/Jadeiin/aitran

# Or pip
pip install git+https://github.com/Jadeiin/aitran
```

Set your API key:

```bash
export AITRAN_API_KEY=sk-...
# or provider-specific:
export OPENAI_API_KEY=sk-...   # OpenAI
export ANTHROPIC_API_KEY=sk-...  # Anthropic
```

## Quick Start

### Agentic workflows

`aitran` launches an agent-driven interactive session. The agent
inspects your project, proposes a plan, and executes each step
(download, translate, review, upload) with your approval.

```bash
# Interactive REPL — multi-turn conversation with session persistence
aitran

# One-shot: give the agent a natural-language request
aitran --prompt "translate Crowdin project MyApp to Chinese"

# Resume a previous session
aitran --resume --session-id abc123

# Use a different orchestrator model
aitran -m openai:gpt-5.4-mini --prompt "translate Weblate component app/zh_Hans"
```

The orchestrator agent uses a separate model (default
`anthropic:claude-sonnet-4-6`) from the translation/review sub-tasks
(default `deepseek:deepseek-v4-flash`), so you can pair a capable
planner with a fast translator.

Write operations (download, translate, review, upload) require explicit
confirmation. Use `--auto-approve` or the `/approve on` REPL command to
skip prompts.

### Direct translation

```bash
# Translate a PO file to its header language
aitran translate --po zh_Hans.po

# Translate with a specific model
aitran translate --po zh_Hans.po -l zh -m anthropic:claude-haiku-4-5

# Translate all PO files in a directory
aitran translate --po-dir ./locales -l zh

# Translate an XLIFF file
aitran translate --xliff en_zh-CN.xliff -src en -l zh

# Verbose mode — see each translation as it completes
aitran translate --po zh_Hans.po -v

# Custom API host (for OpenAI-compatible gateways)
aitran translate --po zh_Hans.po --host https://your-gateway.example.com
```

Models are specified in `<provider>:<model>` format:

- `deepseek:deepseek-v4-flash` (default)
- `openai:gpt-5.4-mini`
- `anthropic:claude-haiku-4-5`
- `anthropic:claude-sonnet-4-5`

## Environment Variables

| Variable | Description |
|---|---|
| `AITRAN_FLOW_MODEL` | Orchestrator model for the top-level interactive app (default: `anthropic:claude-sonnet-4-6`) |
| `AITRAN_FLOW_KEY` | API key for the orchestrator model |
| `AITRAN_FLOW_AUTO_APPROVE` | Auto-approve tools in the interactive app (`1`, `true`, etc.) |
| `AITRAN_API_KEY` / `OPENAI_API_KEY` | API key |
| `AITRAN_API_HOST` | Custom API base URL |
| `AITRAN_MODEL` | Default model (default: `deepseek:deepseek-v4-flash`) |
| `AITRAN_MODEL_TMP` | LLM temperature (default: `0.1`) |
| `AITRAN_LOGFIRE` | Enable Pydantic Logfire tracing (`1`, `true`, etc.) |
| `AITRAN_LOGFIRE_CAPTURE_HTTP` | Capture provider HTTP headers and bodies in Logfire |
| `AITRAN_MLFLOW` | Enable MLflow tracing (`1`, `true`, etc.) |
| `AITRAN_MLFLOW_TRACKING_URI` | MLflow tracking server URI |
| `AITRAN_MLFLOW_EXPERIMENT` | MLflow experiment name |
| `AITRAN_WEBLATE_URL` | Weblate base URL (e.g. `https://weblate.example.org`) |
| `AITRAN_WEBLATE_TOKEN` | Weblate API token |
| `AITRAN_CROWDIN_TOKEN` | Crowdin API token |
| `AITRAN_CROWDIN_ORG` | Crowdin organization (Enterprise only) |
| `AITRAN_CROWDIN_BASE_URL` | Crowdin API base URL override |

## CLI Reference

### app

```
aitran [options]
```

Agentic translation workflow. The agent inspects, downloads, translates,
reviews, and uploads translations on supported localization platforms
with your approval at each write step. If `--prompt` is omitted, `aitran`
starts an interactive REPL with persistent history and session management.

**REPL commands:** `/help`, `/approve on|off|status`, `/resume [id]`, `/exit`

Examples:

```bash
aitran
aitran --prompt "translate Crowdin project MyApp to Chinese"
aitran --resume --session-id abc123
```

Use `aitran --help` for the full list of options.

### translate

```
aitran translate [options]
```

Translate PO or XLIFF files directly without platform integration.

```bash
# Translate a PO file to its header language
aitran translate --po zh_Hans.po

# Translate with a specific model and target language
aitran translate --po zh_Hans.po -l zh -m anthropic:claude-haiku-4-5

# Translate all PO files in a directory
aitran translate --po-dir ./locales -l zh

# Translate an XLIFF file
aitran translate --xliff en_zh-CN.xliff -src en -l zh

# Verbose mode — see each translation as it completes
aitran translate --po zh_Hans.po -v

# Custom API host (for OpenAI-compatible gateways)
aitran translate --po zh_Hans.po --host https://your-gateway.example.com
```

Use `aitran translate --help` for the full list of options.

### Observability

Both `translate` and the top-level interactive app support distributed tracing via Logfire and
MLflow.

**Logfire:**

```bash
uv run logfire auth
uv run logfire projects use  # or: uv run logfire projects new
aitran translate --logfire --po zh_Hans.po
```

Use `--logfire-capture-http` only when you need raw provider requests
and responses in the trace. It may capture prompts, completions,
headers, and API credentials.

**MLflow:**

```bash
aitran translate --mlflow --mlflow-experiment my-project --po zh_Hans.po
```

### sync

```
aitran sync --po <file> --pot <file>
```

Update PO file from POT template, preserving existing translations.

### Platform CLIs

Inspect, download, or upload translation files on supported platforms:

```bash
aitran weblate ls [<project[/component[/lang]]>]
aitran crowdin projects --token <token>
```

Use `aitran weblate --help` or `aitran crowdin --help` for the full
command reference.

### remove

```
aitran remove --po <file> [filters]
```

| Filter | Description |
|---|---|
| `--fuzzy` | Remove fuzzy entries |
| `-obs, --obsolete` | Remove obsolete entries |
| `-ut, --untranslated` | Remove untranslated entries |
| `-t, --translated` | Remove translated entries |
| `-tnf, --translated-not-fuzzy` | Remove translated non-fuzzy entries |
| `-ft, --fuzzy-translated` | Remove fuzzy translated entries |
| `-rc, --reference-contains` | Remove by reference match (plain string or /regex/flags) |

### userdict

```
aitran userdict [-l <lang>] [--explore]
```

Open or explore user dictionaries. Dictionaries are stored in the
platform-standard config directory (XDG-compliant). Name them
`dictionary-<lang>.json` (e.g. `dictionary-zh.json`) and place them
in `.aitran/` under CWD, git root, or the config directory.

Example `dictionary-zh.json`:

```json
{"login": "登录", "logout": "退出", "submit": "提交"}
```

## Features

- **Agentic workflows** — agent-driven translation pipelines with plan-approve-execute pattern
- **Deferred approval** — write operations require explicit confirmation before executing
- **Interactive REPL** — multi-turn conversation with persistent history, auto-suggest, and session management
- **Streaming progress** — translations appear in real time with fuzzy flags
- **Structured output** — Pydantic AI validates completeness, retries on format errors
- **HTML preservation** — XML-escaping from the input format is automatically reversed
- **Fuzzy marking** — uncertain translations are flagged for reviewer attention
- **Translator notes** — optional remarks from the model aid human review
- **XLIFF support** — `needs-review-translation` state for fuzzy units

## Credits

aitran is the successor to the original Node.js gpt-po project that
pioneered the batch translation approach for PO files using LLMs.
aitran carries that vision forward with plans for terminology retrieval,
translation quality review, and multi-agent collaboration.

## License

GPL-2.0-or-later. See [LICENSE](LICENSE).
