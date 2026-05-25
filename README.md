# aitran

AI-powered translation for gettext PO and XLIFF files — streaming
progress, fuzzy flagging, glossary support, and translator notes for
human review. Successor to [gpt-po](https://github.com/ryanhex53/gpt-po).

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
| `AITRAN_API_KEY` / `OPENAI_API_KEY` | API key |
| `AITRAN_API_HOST` | Custom API base URL |
| `AITRAN_MODEL` | Default model (default: `deepseek:deepseek-v4-flash`) |
| `AITRAN_MODEL_TMP` | LLM temperature (default: `0.1`) |
| `AITRAN_LOGFIRE` | Enable Pydantic Logfire tracing (`1`, `true`, etc.) |
| `AITRAN_LOGFIRE_CAPTURE_HTTP` | Capture provider HTTP headers and bodies in Logfire |
| `AITRAN_WEBLATE_URL` | Weblate base URL (e.g. `https://weblate.example.org`) |
| `AITRAN_WEBLATE_TOKEN` | Weblate API token |
| `AITRAN_CROWDIN_TOKEN` | Crowdin API token |
| `AITRAN_CROWDIN_ORG` | Crowdin organization (Enterprise only) |
| `AITRAN_CROWDIN_BASE_URL` | Crowdin API base URL override |

## CLI Reference

### translate

```
aitran translate [options]
```

| Option | Description |
|---|---|
| `-m, --model` | Model in `<provider>:<model>` format |
| `-k, --key` | API key |
| `--host` | Custom API base URL |
| `-t, --temperature` | LLM temperature (default: 0.1) |
| `--po` | PO file path |
| `--po-dir` | Directory of .po files |
| `--xliff` | XLIFF file path |
| `--xliff-dir` | Directory of .xliff/.xlf files |
| `--jobs` | Max files to translate concurrently for directory inputs (default: 4) |
| `-src, --source` | Source language (default: en) |
| `-l, --lang` | Target language (ISO 639-1) |
| `-v, --verbose` | Print each translation as it completes |
| `-o, --output` | Output file path |
| `--context` | Text file with additional translation context |
| `--context-length` | Max accumulated source length per batch (default: 4096) |
| `--order` | Unit ordering: `file` (default), `source`, `reference`, `context` |
| `--profile` | Prompt detail: `full` (default, all metadata) or `fast` (index+source only) |
| `--logfire` | Enable Pydantic Logfire tracing for agent/model runs |
| `--logfire-capture-http` | Capture provider HTTP headers and bodies in Logfire |

### Logfire observability

`aitran translate --logfire ...` enables Pydantic Logfire instrumentation for
Pydantic AI agent/model runs. Set up Logfire first:

```bash
uv run logfire auth
uv run logfire projects use  # or: uv run logfire projects new
```

Use `--logfire-capture-http` only when you need raw provider requests and
responses in the trace. It may capture prompts, completions, headers, and API
credentials, depending on provider/client behavior.

### sync

```
aitran sync --po <file> --pot <file>
```

Update PO file from POT template, preserving existing translations.

### weblate

```
aitran weblate download --url <url> --token <token> --project <slug> --component <slug> -l <lang> -o <file>
aitran weblate upload --url <url> --token <token> --project <slug> --component <slug> -l <lang> --file <file>
```

Download or upload a Weblate translation file for the specified project/component.

### crowdin

```
aitran crowdin download --token <token> --project-id <id> --file-id <id> -l <lang> -o <file>
aitran crowdin upload --token <token> --project-id <id> --file-id <id> -l <lang> --file <file>
```

Download or upload a Crowdin translation file for the specified project file ID.

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
