# Review Agent Design

## Overview

A standalone `aitran review` command that runs translate-toolkit rule-based QA checks + LLM review on already-translated PO/XLIFF files, producing verdicts, corrections, and notes.

## Architecture

### Module Structure (refactored)

```
src/aitran/
  agents/
    __init__.py          # re-exports all public symbols
    _base.py             # build_model(), build_input_xml(), safe_prompt_text(), format_language_label()
    translator.py        # build_translator_agent(), TranslationBatch/TranslatedUnit, SYSTEM_PROMPT/USER_PROMPT
    reviewer.py          # build_reviewer_agent(), ReviewBatch/ReviewedUnit
  qa/
    __init__.py          # QA runner: runs translate-toolkit checkers, returns structured results
    checkers.py          # checker configuration, allowlist
```

Old `agent.py` and `prompts/` module removed — prompts are now inline constants in each agent's file.

### Data Flow

```
aitran review file.po [--strict] [--auto-fix]
    │
    ▼
1. Parse file (PoTranslator / XliffTranslator)
2. Get all translated units
    │
    ▼
3. Run QA checks (translate-toolkit StandardChecker)
   → structured QA errors per unit
    │
    ▼
4. Filter for review:
   - non-strict: only units with QA errors OR fuzzy OR translator notes
   - strict: all units
    │
    ▼
5. Batch by unit count (--batch-size)
6. For each batch:
   - Build input XML: source + target + qa_errors
   - Run reviewer agent → ReviewBatch
    │
    ▼
7. Apply results:
   - pass → no change
   - revise/reject (no auto-fix) → set fuzzy, write note
   - revise/reject + auto-fix + corrected → write target, clear fuzzy, mark translated
    │
    ▼
8. Print summary: pass/revise/reject counts
```

### CLI

```
aitran review (--po file.po | --xliff file.xlf) [options]
  --model             # reuse from translate
  --host              # reuse from translate
  --key               # API key
  --batch-size        # max units per review batch (default: 100)
  --strict            # review all units, not just problematic ones
  --auto-fix          # write corrected targets back to file
```

### Output Schema

```python
class ReviewedUnit(BaseModel):
    index: int
    verdict: str           # "revise" | "reject" (sparse: omitted units implicitly pass)
    corrected: str | None  # corrected target (when reviewer can fix)
    note: str | None       # reason/explanation

class ReviewBatch(BaseModel):
    units: list[ReviewedUnit]
```

### Verdict Semantics

| Verdict | Meaning | Non-auto-fix | Auto-fix |
|---------|---------|-------------|----------|
| *(omitted)* | Translation is OK (sparse output) | No change | No change |
| `revise` | Minor issue, has correction | Set fuzzy + note | Write corrected target, clear fuzzy |
| `reject` + corrected | Serious issue, reviewer can fix | Set fuzzy + note | Write corrected target, clear fuzzy |
| `reject` + no corrected | Serious issue, needs human retranslation | Set fuzzy + note | Set fuzzy + note (target unchanged) |

### Strict / Non-strict Mode

- **Non-strict (default):** Units that pass QA AND have no fuzzy flag AND no translator notes → auto-pass (skip LLM). Only problematic units go to LLM review.
- **Strict:** All units go to LLM review regardless of QA results.

### QA Checks

Broad coverage via translate-toolkit `StandardChecker` — LLM makes final judgment on real vs false positives:

| Layer | Checkers | What |
|-------|----------|------|
| Hard | `printf`, `xmltags`, `xmlentity` | Placeholder/XML integrity |
| Soft | `endpunc`, `brackets`, `caps`, `puncspacing` | Style/consistency |
| Configurable | `accelerators`, `blank`, `dialogs`, etc. | Per-project toggle |

QA errors are fed to the LLM as structured context:

```xml
<unit index="0">
  <source>Hello {name}</source>
  <target>你好</target>
  <qa-errors>missing placeholder {name}</qa-errors>
</unit>
```

### File Write Behavior

- **PO:** `#, fuzzy` for review markers; `# review:` comments for notes; auto-fix writes `msgstr` and clears markers
- **XLIFF:** `markreviewneeded()` for markers; `<note>` for notes; auto-fix uses `settarget()` + `marktranslated()`
- **Pass entries:** Never modified — review only touches problematic units

### Review Prompt

Reviewer agent receives:
- System prompt: role as translation quality reviewer
- Dynamic instructions: language pair, file context, glossary
- Input XML: source + target + qa_errors per unit
- Guidelines: evaluate QA findings (confirm/dismiss), assess overall quality, provide corrections when possible

### Model

Same as translation (`--model`). Review uses fewer tokens (no full target output), so cost is lower.

### Summary Report

Printed to stdout after review:

```
Reviewed: 150 units (23 sent to LLM, 127 auto-passed)
  pass:   120 (80%)
  revise:  25 (17%)
  reject:   5 (3%)
```

## Implementation Phases

1. **Module refactoring:** `agent.py` → `agents/` + `_base.py`
2. **QA module:** `qa/` with translate-toolkit checker integration
3. **Reviewer agent:** `agents/reviewer.py` with prompt, schema, output validator
4. **CLI integration:** `review` subcommand in `cli.py`
5. **File write logic:** PO/XLIFF review marking and auto-fix
6. **Tests:** Unit tests for QA, reviewer agent, file write
