## v0.2.1 (2026-05-25)

### Feat

- apply translate-toolkit api improvements
- redesign prompt context with --profile, --order, and robust HTTP error handling
- update CLI options for PO translation and add concurrent processing support

### Fix

- skip completed xliff source matches
- remove broken default-command pattern, use explicit "aitran translate"
- remove error field from prompt, plumb --order to XLIFF, update README
- update default model to deepseek:deepseek-v4-flash in README and CLI
- remove duplicate reference to OPENAI_API_HOST in environment variables
- update repository URL to github.com/Jadeiin/aitran

### Refactor

- delegate provider dispatch to pydantic-ai's infer_provider_class

## v0.2.0 (2026-05-22)
