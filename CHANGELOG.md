## v0.4.1 (2026-06-01)

### Feat

- add support for eager input streaming in model building
- add support for custom API host and temperature in orchestrator settings
- promote interactive app to top-level CLI

### Refactor

- move app model defaults to cli

## v0.4.0 (2026-05-31)

### Feat

- **flow**: add /resume slash command to restore saved sessions
- **flow**: integrate prompt_toolkit for enhanced REPL UX
- improve orchestrator flow repl
- add orchestrator agent with streaming and platform toolsets
- add full-width bracket support in AITranChecker and corresponding tests
- add observability options for CLI commands and update flush method for MLflow

### Fix

- **flow**: address simplify review findings
- **flow**: keep approval answers out of REPL history
- use get_data() for wlc SDK objects and skip DeferredToolRequests display
- include DeferredToolRequests in orchestrator agent output types
- add skip count to review summary and update related tests
- add summary tracking and save functionality in _run_review_async

### Refactor

- simplify bracket matching logic in _AITranChecker
- rename review_chunk to review_batch and update related logic for batch processing
- simplify v0.3.0 code with dedup, type safety, and QA optimization

## v0.3.0 (2026-05-28)

### Feat

- add MLflow tracing support for pydantic-ai
- add plural support for PoXliff in XliffTranslator and corresponding tests
- enhance review and translation agents with new functionality and improved error handling
- add timeout configuration to HTTP client and corresponding test
- add retrying HTTP client and update dependencies for improved error handling
- add tests for review pipeline and update fuzzy logic in PoTranslator
- implement XML input builders for reviewer and translator agents
- add XLIFF review path to CLI
- add plural form support with unified targets field
- maintain conversation history across review batches
- add review agent with QA + LLM hybrid pipeline
- add legacy language code inference for target language in PoTranslator
- add list and stats handling in Weblate integration with iterator support
- add Weblate and Crowdin commands for listing projects, files, languages, and progress
- update project resolution to use fetch_all for improved data retrieval
- enhance Weblate and Crowdin commands with additional format support and error handling
- enhance Crowdin integration by adding project name support and improving error handling
- enhance Weblate and Crowdin download functionalities with format options
- enhance XML escaping handling in translation process and add related tests
- implement raw markup detection and decoding in translation process
- add name to translator agent
- add optional logfire tracing

### Fix

- pass plural_tags to XLIFF translation pipeline
- update apply_review_batch to use indexed units for PO and XLIFF
- reset HTTP retry counters when skipping failed batch
- decode XML-escaped reviewer corrections before saving
- preserve plural targets when applying review auto-fix
- only combine plural sources for one-form target languages
- detect short plural targets and decode one-form against all sources
- decode each plural target against its corresponding source form
- validate plural target count and decode against all source forms
- remove plural_tags reference from prompt (now in task instructions)
- apply retry logic to final translation batch (P2)
- move review command help text to decorator
- track actual review context consumption across batches
- use report indices in review prompt instead of renumbering
- apply sparse review results by index instead of zip
- revert XML escaping removal and improve prompt guidance
- address latest PR review comments
- address sync helper review comments
- clean up CLI help message
- clean up weblate object handling

### Refactor

- unit-based batching for translate and review
- serial review loop, drop pass verdict, remove logfire spans
- extract review pipeline to review.py with logfire tracing
- use weblate get_object translation flow
- remove unused markup detection logic and simplify entity decoding

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
