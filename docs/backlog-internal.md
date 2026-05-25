# Internal Backlog: Translate Toolkit API Reuse

Last researched: 2026-05-25
Local package version checked in this repo: `translate-toolkit==3.19.10`

This document records engineering opportunities, not task ownership. Use `bd`
for executable issue tracking.

## Sources

- Official API overview: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/index.html
- Storage API reference: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/storage.html
- Storage base class design notes: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/formats/base_classes.html
- Tools API reference: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/tools.html
- Supported formats: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/formats/index.html
- `pofilter` tests: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/commands/pofilter_tests.html
- `pomerge`: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/commands/pomerge.html
- `pocount`: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/commands/pocount.html
- `poterminology`: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/commands/poterminology.html

## Executive Summary

Translate Toolkit is more useful to `aitran` as a Python library than only as a
set of installed CLI commands. The highest-value APIs for this project are:

1. `translate.storage.factory` and the common storage/unit API for generic file
   parsing, serialization, metadata, and unit traversal.
2. `translate.filters.checks.StandardChecker` for post-LLM quality gates.
3. `translate.tools.pocount.calcstats` and `wordsinunit` for estimates and
   progress/cost reporting.
4. `translate.tools.pomerge.mergestores` for safer template/translation merging.
5. `translate.tools.poterminology.TerminologyExtractor` for extracting glossary
   candidates from existing translations.

The main caveat is API maturity. The storage API is usable and already used by
this repo, but some tool modules are command-oriented internals. Prefer small
wrappers with tests over broad direct coupling.

## Current Aitran Usage

`aitran` already depends on Translate Toolkit and currently imports:

- `translate.storage.po` in `translate.py`, `sync.py`, and `manipulate.py`
- `translate.storage.xliff` in `translate.py`

Current adapters duplicate a small shared shape:

- Parse with `po.pofile.parsefile(path)` or `xliff.xlifffile.parsefile(path)`
- Iterate `store.units`
- Read `unit.source`, `unit.target`, `unit.getcontext()`, `unit.getnotes()`
- Decide state through `isheader()`, `istranslated()`, `isfuzzy()`, and XLIFF
  target `state`
- Write `unit.target`, `unit.markfuzzy()`, `unit.addnote()`
- Serialize with `bytes(store)`

That is already close to the common `TranslationStore` / `TranslationUnit`
model, so the next reuse step should be incremental.

## API Findings

### 1. Generic Storage Factory

Relevant API:

```python
from translate.storage import factory

store = factory.getobject(path)
supported = factory.supported_files()
```

Local supported stores reported by `factory.supported_files()`:

- Gettext PO: `po`, `pot`
- XLIFF: `xlf`, `xliff`, `sdlxliff`
- Gettext MO: `mo`, `gmo`
- Qt: `ts`, `qm`, `qph`
- Glossary / memory formats: `tbx`, `tmx`, `utf8`, `tab`, `utx`
- Other localization formats: `catkeys`, `ftl`, `wxl`

Opportunity: create a narrow internal adapter over `TranslationStore` for file
types that expose source/target units cleanly. This would let `translate`
support more storage classes without adding one class per format immediately.

Suggested first slice:

- Keep `PoTranslator` and `XliffTranslator` as special cases.
- Add a shared `StorageTranslator` helper for parse/save/order/common metadata.
- Use `factory.getobject()` only behind explicit CLI options or feature flags at
  first, because not every store has identical translation state semantics.

Risk:

- Some formats are monolingual or converter-oriented. A generic path must verify
  `unit.istranslatable()`, `unit.source`, `unit.target`, and serialization
  before writing in place.

### 2. Built-in QA Checks

Relevant API:

```python
from translate.filters.checks import StandardChecker

checker = StandardChecker()
failures = checker.run_filters(unit, categorised=True)
```

Local smoke test with source `Hello %s` and target `你好` returned failures for:

- `printf`: missing `%s`
- `simplecaps`
- `startcaps`

Useful checks for LLM output:

- Placeholder preservation: `printf`, `pythonbraceformat`, `variables`
- Markup preservation: `xmltags`, `escapes`
- Whitespace/newline preservation: `newlines`, `tabs`, `startwhitespace`,
  `endwhitespace`
- URL/path preservation: `urls`, `filepaths`, `emails`
- Numeric consistency: `numbers`
- Punctuation and casing review hints: `endpunc`, `startpunc`, `simplecaps`

Opportunity: add an optional `--qa` or default post-apply validation pass. For
each translated unit, run selected checks and:

- Mark the unit fuzzy when serious checks fail.
- Add a translator note summarizing failed checks.
- Optionally retry the batch with the QA failure feedback before saving.

Suggested first slice:

- Run a conservative allowlist: `printf`, `pythonbraceformat`, `variables`,
  `xmltags`, `newlines`, `tabs`, `urls`, `filepaths`, `emails`, `numbers`.
- Do not fail the whole command initially; mark fuzzy and annotate.

Risk:

- Some checks are language/style sensitive. Avoid using broad capitalization or
  punctuation checks as hard errors for CJK targets.

### 3. Translation Metrics and Estimation

Relevant API:

```python
from translate.tools import pocount

stats = pocount.calcstats(path)
```

`calcstats()` returns counts for translated, fuzzy, untranslated, review, source
words, target words, and extended states.

Opportunity:

- Add `aitran stats` or `aitran translate --dry-run` to report units/words that
  would be sent to the model.
- Use source word counts for cost estimates and batch planning.
- Report fuzzy/review backlog after translation.

Suggested first slice:

- Implement a read-only CLI command that prints totals for PO/XLIFF through
  `pocount.calcstats()`.

Risk:

- `calcstats()` accepts filenames and uses `factory.getobject()` internally, so
  error handling should be wrapped for user-friendly CLI output.

### 4. Sync and Merge Semantics

Relevant API:

```python
from translate.tools import pomerge

merged = pomerge.mergestores(
    template_store,
    translated_store,
    mergeblanks=False,
    mergefuzzy=True,
    mergecomments=True,
)
```

Current `aitran sync` is PO-specific and manually copies translations from PO to
POT by `(context, msgid)`. `pomerge.mergestores()` already tries `findid()` then
`findunit(source)` and delegates merge behavior to the unit implementation.

Opportunity:

- Replace or augment `sync.py` with `pomerge` semantics.
- Add support for preserving comments/fuzzy flags through documented merge
  options.
- Potentially extend sync behavior to XLIFF or other store types later.

Suggested first slice:

- Add tests comparing current `sync()` behavior with `pomerge.mergestores()` for
  context, fuzzy, comments, obsolete units, and changed source strings.
- Only switch implementation after the exact header/comment behavior is known.

Risk:

- `pomerge` was designed around template/translation merging, not necessarily
  this CLI's current output contract. Header behavior must be pinned by tests.

### 5. Terminology Extraction

Relevant API:

```python
from translate.tools.poterminology import TerminologyExtractor

extractor = TerminologyExtractor(sourcelanguage="en", termlength=3)
extractor.processunits(store.units, fullinputpath=path)
terms = extractor.extract_terms()
```

Opportunity:

- Build glossary candidates from already translated PO/XLIFF files.
- Feed high-confidence term pairs into `.aitran/dictionary-<lang>.json`.
- Offer `aitran userdict --extract-from <po-dir>` or a separate
  `aitran terminology` command.

Suggested first slice:

- Generate a reviewable JSON file, not automatic dictionary mutation.
- Require multiple occurrences before suggesting a term.

Risk:

- Term extraction is heuristic. It should create candidates for human review,
  not authoritative glossary entries.

### 6. Search and Targeted Extraction

Relevant APIs:

- `translate.tools.pogrep.find_matches()`
- `translate.tools.pogrep.rungrep()`

Opportunity:

- Improve `aitran remove --reference-contains` and future search commands by
  reusing Toolkit search semantics across source, target, notes, and locations.
- Add a targeted retranslation workflow: extract matching units, translate them,
  then merge back through `pomerge`.

Suggested first slice:

- Keep current `remove` implementation for now.
- Consider `pogrep` when adding a non-destructive `aitran grep` or
  `aitran extract` command.

Risk:

- `rungrep()` is file-oriented and writes an output store. Direct unit-level use
  may require more wrapper code than the current regex filter.

### 7. Converter Ecosystem

Translate Toolkit ships many converters, including Markdown, MDX, JSON, YAML,
TOML, Fluent, Android resources, RESX, Qt TS, HTML, subtitles, and more.

Opportunity:

- Use converter commands as an external pipeline: source format -> PO/XLIFF ->
  `aitran` -> source format.
- This could make `aitran` useful beyond PO/XLIFF without implementing every
  storage format directly.

Suggested first slice:

- Document manual workflows first, for example `md2po`, `aitran translate`,
  then `po2md`.
- Add automated wrappers only for formats with strong round-trip tests.

Risk:

- Round-trip fidelity varies by format. The first automated wrapper should use
  sample fixtures that include placeholders, markup, comments, and ordering.

### 8. Language Metadata

Relevant API:

```python
from translate.lang import factory as lang_factory

lang = lang_factory.getlanguage("en")
```

Available local language object behavior includes language names, plurals,
sentence/word splitting, punctuation translation, and checker preferences.

Opportunity:

- Improve target language display in prompts and CLI output.
- Set or validate PO plural headers through language metadata.
- Use language word/sentence splitting for better batch boundaries.

Suggested first slice:

- Use language metadata only for display and diagnostics.
- Avoid automatic plural header rewriting until tested across target languages.

Risk:

- Language metadata may be incomplete for some locale variants.

## Recommended Backlog Order

1. Add post-translation QA checks using `StandardChecker` and mark fuzzy with
   notes on serious failures.
2. Add `aitran stats` / `--dry-run` using `pocount.calcstats()`.
3. Prototype `pomerge`-backed sync behind tests.
4. Add glossary candidate extraction using `TerminologyExtractor`.
5. Prototype generic storage parsing through `factory.getobject()` for one
   additional format with reliable source/target semantics.
6. Document converter round-trip workflows before automating them.

## Verification Commands Used

```bash
uv run python -c "import importlib.metadata as m; print(m.version('translate-toolkit'))"
uv run python -c "from translate.storage import factory; print(factory.supported_files())"
uv run python -c "from translate.filters.checks import StandardChecker; print(StandardChecker().getfilters())"
uv run python -c "from translate.tools import pocount; import inspect; print(inspect.signature(pocount.calcstats))"
```
