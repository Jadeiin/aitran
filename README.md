# aitran

Agentic translation workflows for localization platforms, plus direct
PO/XLIFF translation with bounded tool retrieval, deferred approval, and
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

`aitran` starts Pydantic AI's WebUI with the workflow orchestrator. The agent
inspects your localization project, proposes a plan, and delegates each step
(download, translate, review, upload) with your approval in the browser.

```bash
# Start the WebUI at http://127.0.0.1:7932
aitran

# Bind the WebUI to another address and port
aitran -m openai:gpt-5.4-mini --web-host 0.0.0.0 --web-port 9000
```

The orchestrator uses Pydantic AI Harness `SubAgents` to delegate isolated,
self-contained tasks directly to the translator and reviewer. The orchestrator,
translator, and reviewer can use separate models; reviewer settings fall back to
translator settings when omitted.

Write operations (download, translation delegation, review delegation, upload)
use Pydantic AI's native deferred-tool approval in the WebUI. Translator and
reviewer subagents retrieve their assigned PO/XLIFF batches through tools;
translation entries are not injected into the orchestrator conversation.

Headless `translate` and `review` commands consume the same native Agent tool
events and render one Rich progress row per file as batches are written.

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

# Verbose mode — print the completed artifact JSON
aitran translate --po zh_Hans.po -v

# Custom API host (for OpenAI-compatible gateways)
aitran translate --po zh_Hans.po --host https://your-gateway.example.com

# Free-form product/domain guidance is separate from the fast/full metadata profile
aitran translate --po zh_Hans.po --instructions "Mobile UI; keep product names"
```

Models are specified in `<provider>:<model>` format:

- `deepseek:deepseek-v4-flash` (default)
- `openai:gpt-5.4-mini`
- `anthropic:claude-haiku-4-5`
- `anthropic:claude-sonnet-4-5`

## Environment Variables

| Variable | Description |
|---|---|
| `AITRAN_APP_MODEL` | Orchestrator model for the WebUI (default: `deepseek:deepseek-v4-pro`) |
| `AITRAN_APP_KEY` | API key for the orchestrator model |
| `AITRAN_APP_HOST` | Custom API base URL for the orchestrator model |
| `AITRAN_APP_TMP` | Orchestrator LLM temperature (default: `0.5`) |
| `AITRAN_TRANSLATOR_MODEL` | Translator model (falls back to `AITRAN_MODEL`) |
| `AITRAN_TRANSLATOR_KEY` | Translator API key (falls back to `AITRAN_API_KEY`) |
| `AITRAN_TRANSLATOR_HOST` | Translator API base URL (falls back to `AITRAN_API_HOST`) |
| `AITRAN_TRANSLATOR_TMP` | Translator temperature (falls back to `AITRAN_MODEL_TMP`) |
| `AITRAN_REVIEWER_MODEL` | Reviewer model (falls back to translator model) |
| `AITRAN_REVIEWER_KEY` | Reviewer API key (falls back to translator key) |
| `AITRAN_REVIEWER_HOST` | Reviewer API base URL (falls back to translator host) |
| `AITRAN_REVIEWER_TMP` | Reviewer temperature (falls back to translator temperature) |
| `AITRAN_WEB_HOST` | WebUI bind host (default: `127.0.0.1`) |
| `AITRAN_WEB_PORT` | WebUI bind port (default: `7932`) |
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

Starts the orchestrator WebUI. The agent inspects, downloads, delegates
translation and review to specialized subagents, and uploads translations on
supported localization platforms with native approval controls at each write
step.

Examples:

```bash
aitran
aitran --web-host 0.0.0.0 --web-port 9000
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

Both `translate` and the top-level WebUI support distributed tracing via Logfire and
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
- **Native WebUI** — multi-turn orchestration through Pydantic AI's built-in web frontend
- **Bounded retrieval** — translator/reviewer Agents pull pages by both unit and character budgets
- **Headless progress** — concurrent files update independent Rich progress rows without polluting artifact output
- **Structured tool contracts** — Pydantic validates batch identity, completeness, plural counts, and artifact paths
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
