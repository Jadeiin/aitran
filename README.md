# aitran

AI-powered translation for gettext PO and XLIFF files — streaming
progress, fuzzy flagging, glossary support, and translator notes for
human review. Successor to [gpt-po](https://github.com/ryanhex53/gpt-po).

Supports OpenAI, Anthropic, DeepSeek, and any OpenAI-compatible provider.

## Installation

```bash
# Recommended: uv
uv tool install aitran

# Or pip
pip install aitran
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
aitran --po zh_Hans.po

# Translate with a specific model
aitran --po zh_Hans.po -l zh -m anthropic:claude-haiku-4-5

# Translate all PO files in a directory
aitran --dir ./locales -l zh

# Translate an XLIFF file
aitran --xliff en_zh-CN.xliff -src en -l zh

# Verbose mode — see each translation as it completes
aitran --po zh_Hans.po -v

# Custom API host (for OpenAI-compatible gateways)
aitran --po zh_Hans.po --host https://your-gateway.example.com
```

Models are specified in `<provider>:<model>` format:

- `deepseek:deepseek-chat` (default)
- `openai:gpt-5.4-mini`
- `anthropic:claude-haiku-4-5`
- `anthropic:claude-sonnet-4-5`

## Environment Variables

| Variable | Description |
|---|---|
| `AITRAN_API_KEY` / `OPENAI_API_KEY` | API key |
| `AITRAN_API_HOST` | Custom API base URL |
| `AITRAN_MODEL` | Default model (default: `deepseek:deepseek-chat`) |
| `AITRAN_MODEL_TMP` | LLM temperature (default: `0.1`) |

## CLI Reference

### translate (default command)

```
aitran [options]
```

| Option | Description |
|---|---|
| `-m, --model` | Model in `<provider>:<model>` format |
| `-k, --key` | API key |
| `--host` | Custom API base URL |
| `-t, --temperature` | LLM temperature (default: 0.1) |
| `--po` | PO file path |
| `--dir` | Directory of .po files |
| `--xliff` | XLIFF file path |
| `--xliff-dir` | Directory of .xliff/.xlf files |
| `-src, --source` | Source language (default: en) |
| `-l, --lang` | Target language (ISO 639-1) |
| `-v, --verbose` | Print each translation as it completes |
| `-o, --output` | Output file path |
| `--context` | Text file with additional translation context |
| `--context-length` | Max accumulated source length per batch (default: 4096) |

### sync

```
aitran sync --po <file> --pot <file>
```

Update PO file from POT template, preserving existing translations.

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

[WTFPL](http://www.wtfpl.net/) — Do What the Fuck You Want to Public License.
