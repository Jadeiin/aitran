# Internal Backlog: Translate Toolkit API Reuse

Last researched: 2026-05-25
Local package version checked in this repo: `translate-toolkit==3.19.10`

This document records engineering opportunities, not task ownership. Use `bd`
for executable issue tracking.

## Scope And Caveats

Translate Toolkit is not just the `po*` command family. Its documented API is
organized around:

- `storage`: storage classes for localization formats
- `filters`: translation checks, autocorrection, and string decoration helpers
- `lang`: language-specific metadata and text processing
- `search`: translation memory and terminology matching
- `convert`: conversion modules between localization/document formats
- `misc`: quoting, XML, case-insensitive dictionaries, multistrings, data files
- `tools`: higher-level operations, mostly PO-focused

The official "latest" docs currently render mixed page versions in places
(`3.16.x` to `3.19.x`), while this repo has `3.19.10` installed. The inventory
below therefore combines official docs with local runtime inspection.

Useful official references:

- API overview: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/index.html
- Storage API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/storage.html
- Filters API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/filters.html
- Convert API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/convert.html
- Lang API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/lang.html
- Search API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/search.html
- Misc API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/misc.html
- Storage base class notes: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/formats/base_classes.html
- Supported formats: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/formats/index.html
- Tools API: https://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/tools.html

## What Aitran Already Uses

Current imports:

- `translate.storage.po` in `translate.py`, `sync.py`, `manipulate.py`
- `translate.storage.xliff` in `translate.py`

The current adapters already rely on the common storage/unit shape:

- stores: `parsefile()`, `units`, `gettargetlanguage()`, `updateheader()`,
  `bytes(store)`
- units: `source`, `target`, `getcontext()`, `getnotes()`, `getlocations()`,
  `isheader()`, `istranslated()`, `isfuzzy()`, `markfuzzy()`, `addnote()`

That means the main reuse opportunity is not "call more tools"; it is to build
a small internal compatibility layer over `TranslationStore` and
`TranslationUnit`.

## Dependency Reality

`translate-toolkit` exposes many optional integrations through extras. Current
`aitran` only depends on `translate-toolkit>=3.14`, so several modules fail to
import locally unless extras are added.

Installed base dependencies:

- `lxml`
- `unicode-segmentation-rs`

Optional extras visible from package metadata:

- `chardet`: `charset-normalizer`
- `fluent`: `fluent.syntax`
- `ical`: `vobject`
- `ini`: `iniparse`
- `levenshtein`: `RapidFuzz`
- `markdown`: `mistletoe`
- `php`: `phply`
- `rc`: `pyparsing`
- `spellcheck`: `pyenchant`
- `subtitles`: `aeidon`
- `toml`: `tomlkit`
- `yaml`: `ruamel.yaml`

Observed local import failures:

- `translate.storage.fluent`: missing `fluent.syntax`
- `translate.storage.yaml`: missing `ruamel`
- `translate.storage.markdown`: missing `mistletoe`
- `translate.convert.md2po` / `po2md`: missing `mistletoe`
- `translate.convert.yaml2po` / `po2yaml`: missing `ruamel`
- `translate.search.match`: works, but warns that `RapidFuzz` is missing and
  falls back to slower built-in fuzzy matching

Backlog implication: do not promise Markdown/YAML/Fluent direct support unless
we either add extras or shell out to an environment that has them.

## Storage API

Core API:

```python
from translate.storage import factory

store = factory.getobject(path)
supported = factory.supported_files()
```

Common store methods from `translate.storage.base.TranslationStore`:

- parse/serialize: `parse()`, `parsefile()`, `parsestring()`, `serialize()`,
  `save()`, `savefile()`
- units: `units`, `unit_iter()`, `getunits()`, `addunit()`, `addsourceunit()`,
  `removeunit()`
- lookup/indexing: `makeindex()`, `require_index()`, `findid()`, `findunit()`,
  `findunits()`, `getids()`
- language/project metadata: `getsourcelanguage()`, `setsourcelanguage()`,
  `gettargetlanguage()`, `settargetlanguage()`, `getprojectstyle()`,
  `setprojectstyle()`, `get_plural_tags()`, `get_base_locale_code()`
- merge/translation helpers: `merge_on`, `translate()`,
  `suggestions_in_format`

Common unit methods from `translate.storage.base.TranslationUnit`:

- identity/context: `getid()`, `setid()`, `getcontext()`, `setcontext()`,
  `getdocpath()`, `setdocpath()`, `rid`, `xid`, `line_number`
- text: `source`, `target`, `rich_source`, `rich_target`,
  `multistring_to_rich()`, `rich_to_multistring()`
- locations/notes/errors: `getlocations()`, `addlocation()`, `addlocations()`,
  `getnotes()`, `addnote()`, `removenotes()`, `geterrors()`, `adderror()`
- state: `istranslatable()`, `istranslated()`, `isblank()`, `isheader()`,
  `isfuzzy()`, `isreview()`, `isobsolete()`, `markfuzzy()`,
  `markreviewneeded()`, `marktranslatable()`, `makeobsolete()`,
  `get_state_n()`, `set_state_n()`, `infer_state()`
- merge/previous metadata: `merge()`, `buildfromunit()`, `set_as_previous()`,
  `copy_previous()`, `clear_previous()`, `has_previous()`, `prev_source`,
  `prev_target`, `prev_context`, `getalttrans()`
- plurals: `hasplural()`, `sync_plural_count()`, `gettargetlen()`

Local `factory.supported_files()` returns:

- Gettext PO/POT: `po`, `pot`
- XLIFF: `xlf`, `xliff`, `sdlxliff`
- Gettext MO/GMO: `mo`, `gmo`
- Qt: `ts`, `qm`, `qph`
- Glossary / memory: `tbx`, `tmx`, `utf8`, `tab`, `utx`
- Other direct storage: `catkeys`, `ftl`, `wxl`

Additional installed storage modules include many converter-facing formats:
`asciidoc`, `csvl10n`, `dtd`, `flatxml`, `html`, `ical`, `idml`, `ini`,
`jsonl10n`, `markdown`, `mdxfile`, `mozilla_lang`, `php`, `properties`, `rc`,
`resx`, `stringsdict`, `subtitles`, `toml`, `yaml`, and more. These are not all
registered as direct `factory.supported_files()` stores.

### Storage Opportunities

1. Introduce an internal `UnitAdapter` protocol.

   It should wrap only the subset `aitran` needs:
   `source`, `target`, `context`, `notes`, `locations`, `istranslatable`,
   `istranslated`, `isfuzzy`, `mark_fuzzy`, `add_note`, `set_target`.

   This gives us a stable seam over Toolkit's broad API without treating every
   format as equivalent.

2. Use `getdocpath()` for stable context where available.

   Current ordering supports `reference` and `context`. `getdocpath()` can give
   document-structure context for HTML/XML/document-derived units where line
   numbers are unstable.

3. Preserve previous-source metadata.

   `prev_source`, `prev_target`, `prev_context`, `has_previous()`, and
   `getalttrans()` can feed better prompts for fuzzy or migrated units.

4. Use `rich_source` / `rich_target` carefully for inline-code aware prompts.

   Toolkit's rich string/placeables model may expose tags/placeholders more
   reliably than treating everything as escaped XML text. This is worth a
   prototype for XLIFF and HTML-derived units.

5. Prefer direct storage for bilingual formats, converter pipeline for document
   formats.

   Good direct candidates: PO, XLIFF, Qt TS, TMX/TBX/QPH/UTX/OmegaT glossary,
   RESX, properties, stringsdict, catkeys, WXL.

   Riskier direct candidates: Markdown/YAML/TOML/HTML/ODF-like document formats.
   These should first be treated as convert-to-PO/XLIFF pipelines with
   round-trip fixtures.

## Format-Specific Notes

### PO / POT

Strong API and already used.

Extra things worth using:

- `pofile.getheaderplural()`, `updateheaderplural()`, `settargetlanguage()`
- `pounit.hasplural()`, `sync_plural_count()`, multistring targets
- `pounit.copy_previous()`, `clear_previous()`
- `pofile.removeduplicates()` for cleanup workflows

Backlog:

- Make plural handling explicit in prompts and tests.
- Use language metadata to validate PO plural headers before translation.

### XLIFF 1.x / SDLXLIFF

Already used, but current adapter edits XML directly for target state. Toolkit
has higher-level methods:

- `xliffunit.gettarget()`, `settarget()`
- `marktranslated()`, `markreviewneeded()`, `markapproved()`, `isapproved()`
- `getcontextgroups()`, `getcontextgroupsbyattribute()`
- `source_dom`, `target_dom`, `get_source_dom()`, `set_target_dom()`
- alternate translations through `addalttrans()`, `getalttrans()`

Backlog:

- Replace direct target `state` XML edits with unit methods where behavior
  matches our output contract.
- Feed XLIFF context groups into prompts.
- Preserve or generate `alt-trans` suggestions for review workflows.

### TMX / Glossary Formats

Relevant modules/classes:

- `translate.storage.tmx.tmxfile`, `tmxunit`
- `translate.storage.tbx.tbxfile`, `tbxunit`
- `translate.storage.qph.QphFile`
- `translate.storage.omegat.OmegaTFile`
- `translate.storage.utx.UtxFile`

Backlog:

- Import translation memories and glossaries as prompt context.
- Export accepted translations into TMX/QPH/TBX rather than only `.aitran`
  JSON dictionaries.
- Add a pluggable glossary provider interface so JSON dictionaries and Toolkit
  glossary stores can share the same lookup pipeline.

### JSON / Properties / RESX / Strings

Relevant modules:

- `translate.storage.jsonl10n`
- `translate.storage.properties`
- `translate.storage.resx`
- `translate.storage.stringsdict`

Backlog:

- Prototype direct parse/translate/save for one key-value format, probably
  RESX or Java properties, before promising generic JSON.
- Preserve keys as `context` or `docpath` in prompts.

Risk:

- JSON has multiple dialect classes (`I18Next`, `FormatJS`, `GoI18N`, ARB,
  Nextcloud, WebExtension). Generic JSON support needs dialect selection.

## Filters API

### Checks

Core API:

```python
from translate.filters.checks import StandardChecker

checker = StandardChecker()
failures = checker.run_filters(unit, categorised=True)
```

Useful `StandardChecker` tests for LLM output:

- placeholders: `printf`, `pythonbraceformat`, `variables`
- markup/escapes: `xmltags`, `escapes`
- whitespace/layout: `newlines`, `tabs`, `startwhitespace`, `endwhitespace`
- references: `urls`, `filepaths`, `emails`
- data consistency: `numbers`, `acronyms`
- review-only style hints: `endpunc`, `startpunc`, `simplecaps`,
  `doublewords`, `doublespacing`

Backlog:

- Add a `QualityChecker` wrapper with a conservative default allowlist.
- If serious checks fail, mark fuzzy and add a translator note.
- Later, add retry-with-QA-feedback for batches that violate hard checks.

Risk:

- Broad style checks are language-sensitive. For CJK targets, capitalization and
  punctuation checks should be warnings at most.

### Autocorrect

Core API:

```python
from translate.filters import autocorrect

target = autocorrect.correct(source, target)
```

Docs describe automatic fixes for ellipsis consistency, missing leading/trailing
whitespace, and missing terminal punctuation.

Backlog:

- Add an opt-in `--autocorrect` postprocessor after model output and before QA.
- Keep the original model output in notes if autocorrect changed text.

Risk:

- Autocorrect may be too aggressive for UI copy or CJK punctuation conventions.

### Decoration / Prefilters / Helpers

Relevant APIs:

- `decoration.getvariables()`, `findmarkedvariables()`
- `decoration.getnumbers()`, `geturls()`, `getemails()`, `getfunctions()`
- `decoration.findaccelerators()`, `countaccelerators()`,
  `isvalidaccelerator()`
- `prefilters.filteraccelerators()`, `filtervariables()`,
  `removekdecomments()`
- `helpers.countmatch()`, `funcmatch()`, `multifilter()`

Backlog:

- Extract placeholders, variables, URLs, emails, and accelerators before
  prompting and include them as explicit preservation constraints.
- Use accelerator helpers for UI formats where `&File` / `_File` style access
  keys matter.

## Lang API

Core APIs:

```python
from translate.lang import factory as lang_factory
from translate.lang import data as lang_data

lang = lang_factory.getlanguage("zh_CN")
lang_data.normalize_code("zh-CN")
lang_data.simplify_to_common("zh_Hans_CN")
lang_data.is_rtl("ar")
```

Useful language object behavior observed locally:

- names/codes: `code`, `fullname`
- plurals: `nplurals`, `pluralequation`
- tokenization: `sentences()`, `words()`
- punctuation and number transforms: `punctranslate()`, `numbertranslate()`
- QA preferences: `checker`, `ignoretests`
- heuristics: `length_difference()`, `capsstart()`

Observed examples:

- `zh_CN`: `nplurals=1`, `pluralequation=0`, fullname `Chinese (China)`
- `ja`: `nplurals=1`, `pluralequation=0`
- `ar`: `nplurals=6`, Arabic plural expression
- `ru`: `nplurals=3`, Russian plural expression

Backlog:

- Normalize `--lang` and PO/XLIFF language codes through `lang_data`.
- Display language names in prompts: "Chinese (China)" is clearer than `zh_CN`.
- Use `nplurals` / `pluralequation` to validate PO plural headers.
- Use source `sentences()` for smarter batch boundaries.
- Use `is_rtl()` to warn about RTL target formats and bidirectional marks.

Risk:

- Some language modules are sparse. Treat language metadata as advisory unless
  tests prove it for target locales we support.

## Search API

Relevant APIs:

```python
from translate.search.match import matcher, terminologymatcher

tm = matcher(store, max_candidates=10, min_similarity=75)
matches = tm.matches("source text")

terms = terminologymatcher(store)
term_matches = terms.matches("long source text")
```

Other relevant classes:

- `translate.search.lshtein.LevenshteinComparer`
- `translate.search.terminology.TerminologyComparer`

Backlog:

- Build a translation-memory provider from existing translated stores.
- Before sending a unit to the model, find fuzzy matches and include top
  source/target examples in the prompt.
- Use `terminologymatcher` for inline term suggestions from glossary stores.
- Add `RapidFuzz` extra if matching is used in hot paths.

Risk:

- Fuzzy TM matches can reinforce stale translations. Prompt should label them
  as examples, not required output.

## Convert API

There are two levels:

1. `translate.convert.factory` has a generic `convert()` function and registry
   concept.
2. Individual converter modules expose direct functions such as
   `json2po.convertjson()` and `po2json.convertjson()`.

Important local finding: `translate.convert.factory.converters` is empty in the
installed package. So `factory.convert()` is not currently useful unless
converters are registered by the caller. Direct converter modules are more
practical.

Relevant direct converter functions inspected locally:

- `translate.convert.json2po.convertjson(...)`
- `translate.convert.po2json.convertjson(...)`
- `translate.convert.po2xliff.convertpo(...)`
- `translate.convert.xliff2po.convertxliff(...)`
- `translate.convert.po2tmx.convertpo(...)`
- `translate.convert.ts2po.convertts(...)`
- `translate.convert.po2ts.convertpo(...)`
- `translate.convert.resx2po.convert_resx(...)`
- `translate.convert.po2resx.convertresx(...)`
- `translate.convert.html2po.converthtml(...)`
- `translate.convert.po2html.converthtml(...)`

Optional-extra converter failures observed:

- `md2po` / `po2md` require `mistletoe`
- `yaml2po` / `po2yaml` require `ruamel.yaml`

Backlog:

- Do not build on `convert.factory.convert()` yet.
- For each source format, wrap the direct converter function and pin behavior
  with round-trip fixtures.
- Start with JSON, RESX, TS, HTML, or properties-style formats before Markdown
  and YAML.

Risk:

- Converter APIs are less stable and more command-shaped than storage base
  classes. Wrappers should be thin and heavily tested.

## Misc API

Useful modules:

### `translate.misc.multistring`

Represents plural or multi-part strings. This matters for PO plurals and some
storage formats. Current `aitran` mostly treats source/target as plain strings;
plural-aware work should use `multistring` rather than ad hoc lists.

### `translate.misc.quote`

Useful helpers:

- `htmlentitydecode()`, `htmlentityencode()`
- `propertiesdecode()`, `javapropertiesencode()`,
  `java_utf8_properties_encode()`
- `escapecontrols()`, `rstripeol()`
- `extract()`, `extractwithoutquotes()`

Backlog:

- Replace any future format-specific escaping code with Toolkit helpers when
  the format matches.
- Review current `html.unescape()` usage against Toolkit quote helpers.

### `translate.misc.xml_helpers`

Useful helpers:

- `get_safe_xml_parser()`, `parse_xml()`, `parse_xml_file()`
- `getXMLlang()`, `setXMLlang()`, `getXMLspace()`, `setXMLspace()`
- `normalize_xml_space()`, `normalize_space()`
- `valid_chars_only()`, `safely_set_text()`
- `namespaced()`, `reindent()`

Backlog:

- Use these for future XLIFF/XML mutations instead of direct lxml boilerplate.
- Use `valid_chars_only()` before writing model output into XML-backed formats.

### `translate.misc.dictutils.cidict`

Case-insensitive dictionary. Could be useful for glossary matching where source
terms should be case-insensitive but original keys must remain accessible.

### `translate.misc.file_discovery`

`get_abs_data_filename()` can locate Toolkit data files such as stoplists used
by terminology extraction.

## Tools API

Tools are still useful, but they should be a secondary layer over library APIs.

Useful modules:

- `translate.tools.pocount.calcstats(path)` and `wordsinunit(unit)` for counts
- `translate.tools.pomerge.mergestores()` for merge semantics
- `translate.tools.pogrep.find_matches()` / `rungrep()` for extraction/search
- `translate.tools.poterminology.TerminologyExtractor` for glossary candidates
- `translate.tools.pretranslate` for TM-based pretranslation workflows
- `translate.tools.posegment` for segmentation workflows

Backlog:

- `stats` / `dry-run`: use `pocount.calcstats()` or storage `Statistics`.
- `sync`: compare current manual PO sync against `pomerge.mergestores()`.
- `terminology`: use `TerminologyExtractor` to generate reviewable dictionary
  candidates, not automatic glossary writes.
- `pretranslate`: inspect separately before using; it may overlap strongly with
  the Search API.

## Recommended Backlog Order

1. Build a small `TranslationStore` / `TranslationUnit` adapter layer and move
   PO/XLIFF onto it without changing behavior.
2. Add conservative post-translation QA with `StandardChecker`, marking fuzzy
   and adding notes on hard failures.
3. Add language normalization and plural-header validation through `lang`.
4. Add `stats` / `dry-run` based on `pocount` or `storage.statistics`.
5. Add TM/glossary providers using `search.matcher`, `terminologymatcher`, and
   TMX/TBX/QPH/UTX/OmegaT stores.
6. Prototype `pomerge`-backed sync behind compatibility tests.
7. Prototype one new direct storage format, with RESX or Qt TS as better first
   candidates than Markdown/YAML.
8. Add converter-pipeline support format by format, using direct converter
   modules and round-trip fixtures.

## Verification Commands Used

```bash
uv run python -c "import importlib.metadata as m; print(m.version('translate-toolkit'))"
uv run python -c "from translate.storage import factory; print(factory.supported_files())"
uv run python -c "from translate.storage import base; print(dir(base.TranslationStore)); print(dir(base.TranslationUnit))"
uv run python -c "from translate.filters.checks import StandardChecker; print(StandardChecker().getfilters())"
uv run python -c "from translate.lang import factory; print(factory.getlanguage('zh_CN').nplurals)"
uv run python -c "from translate.search.match import matcher; print(dir(matcher))"
uv run python -c "import importlib.metadata as m; print(m.distribution('translate-toolkit').metadata.get_all('Requires-Dist'))"
```
