"""Core translation engine with multi-turn conversation support."""

import os
import time
from importlib.resources import files

import litellm
from litellm.exceptions import RateLimitError, Timeout
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from translate.storage import po, xliff

from aitran.dicts import find_matching_entries
from aitran.prompts import (
    ParseError,
    StreamParser,
    UnitProtocol,
    format_batch_xml,
    load_system_prompt,
    load_user_prompt,
    parse_translations,
)
def _read_context(context_file: str | None) -> str:
    if not context_file:
        return ""
    with open(context_file, encoding="utf-8") as f:
        return f.read().strip()


def _get_api_key() -> str | None:
    return os.environ.get("AITRAN_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _get_api_host() -> str | None:
    return os.environ.get("AITRAN_API_HOST") or os.environ.get("OPENAI_API_HOST")


def _get_temperature() -> float:
    val = os.environ.get("AITRAN_MODEL_TMP") or os.environ.get("OPENAI_MODEL_TMP")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return 0.1


class TranslationSession:
    """Manages a multi-turn conversation for translating one file.

    The session holds an accumulating messages list so that system prompt,
    guidelines, and dictionary entries are sent only once. Subsequent batches
    benefit from prompt caching (Anthropic explicit, OpenAI automatic).
    """

    def __init__(self, model: str, timeout: int = 20000) -> None:
        self.model = model
        self.timeout = timeout
        self.messages: list[dict] = []
        self._idx_offset = 0
        self._total_units = 0  # running count of units translated so far

    def setup(
        self,
        system_prompt: str,
        user_prompt: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        dict_entries: list[tuple[str, str]] | None = None,
    ) -> None:
        """Populate preamble messages (Turns 1-3).

        After setup, the preamble is eligible for prompt caching on subsequent
        API calls.
        """
        # Turn 1: system prompt
        sys_content = system_prompt
        if context:
            sys_content += f"\n\nContext: {context}"
        self.messages.append({"role": "system", "content": sys_content})

        # Turn 2: guidelines + task description
        task_desc = (
            f"\n\nWait for my incoming message(s) in `{source_lang}` and "
            f"translate them into `{target_lang}` (`{source_lang}` and "
            f"`{target_lang}` are XPG/POSIX locale names, used in Unix-like "
            f"systems and GNU Gettext)."
        )
        self.messages.append({"role": "user", "content": user_prompt + task_desc})
        self.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Understood, I will translate your incoming "
                    f"`{source_lang}` message(s) into `{target_lang}`, "
                    f"carefully following guidelines. Please go ahead and "
                    f"send your message(s) for translation."
                ),
            }
        )

        # Turn 3: dictionary entries (if any)
        if dict_entries:
            user_entries: list[str] = []
            asst_entries: list[str] = []
            for i, (key, val) in enumerate(dict_entries, start=1):
                user_entries.append(f'<translate index="{i}">{key}</translate>')
                asst_entries.append(f'<translated index="{i}">{val}</translated>')
            self.messages.append({"role": "user", "content": "\n".join(user_entries)})
            self.messages.append(
                {"role": "assistant", "content": "\n".join(asst_entries)}
            )
            self._idx_offset = len(dict_entries)

    def translate_batch(
        self,
        units: list[UnitProtocol],
        on_progress=None,  # callable(src: str, translation: str) | None
    ) -> list[str]:
        """Send one batch of source strings and return parsed translations.

        When on_progress is provided, the batch is streamed and the callback
        is invoked for each translation unit as it completes.
        """
        start_index = self._idx_offset + self._total_units + 1
        batch_xml = format_batch_xml(units, start_index)
        self.messages.append({"role": "user", "content": batch_xml})

        kwargs: dict = {
            "model": self.model,
            "messages": self.messages,
            "temperature": _get_temperature(),
            "timeout": self.timeout / 1000,  # litellm uses seconds
        }

        api_key = _get_api_key()
        if api_key:
            kwargs["api_key"] = api_key

        api_host = _get_api_host()
        if api_host:
            kwargs["api_base"] = api_host.rstrip("/") + "/v1"

        model_lower = self.model.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            kwargs["cache_control_injection_points"] = [
                {"location": "message", "role": "system"}
            ]

        if on_progress:
            return self._translate_batch_streaming(
                kwargs, start_index, units, on_progress
            )

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})

        translations = parse_translations(content, start_index, len(units))
        self._total_units += len(units)
        return translations

    def _translate_batch_streaming(
        self,
        kwargs: dict,
        start_index: int,
        units: list[UnitProtocol],
        on_progress,
    ) -> list[str]:
        """Stream the batch response, calling on_progress per completed unit."""
        kwargs["stream"] = True
        response = litellm.completion(**kwargs)

        buffer = ""
        parser = StreamParser(start_index, len(units))

        for chunk in response:
            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue
            buffer += delta.content
            parser.feed(delta.content)

            for idx, text in parser.newly_completed:
                local_idx = idx - start_index
                if 0 <= local_idx < len(units):
                    on_progress(units[local_idx].source, text)

        self.messages.append({"role": "assistant", "content": buffer})

        translations = parser.get_result()
        self._total_units += len(units)
        return translations


class PoTranslator:
    """Handles PO file parsing, filtering, and output."""

    @staticmethod
    def parse(path: str) -> po.pofile:
        return po.pofile.parsefile(path)

    @staticmethod
    def get_header_language(po_file: po.pofile) -> str | None:
        lang = po_file.gettargetlanguage()
        return lang if lang else None

    @staticmethod
    def get_untranslated(po_file: po.pofile) -> list[po.pounit]:
        """Return units that need translation (empty target or fuzzy)."""
        result: list[po.pounit] = []
        for unit in po_file.units:
            if unit.isheader():
                continue
            if unit.istranslated() and not unit.isfuzzy():
                continue
            result.append(unit)
        return result

    @staticmethod
    def trim_fuzzy_targets(po_file: po.pofile) -> bool:
        """Trim leading/trailing spaces from translated targets. Returns True if any changed."""
        changed = False
        for unit in po_file.units:
            if unit.isheader():
                continue
            if unit.target and isinstance(unit.target, str):
                trimmed = unit.target.strip()
                if trimmed != unit.target:
                    unit.target = trimmed
                    changed = True
        return changed

    @staticmethod
    def apply_batch(
        po_file: po.pofile, units: list[po.pounit], translations: list[str]
    ) -> None:
        for unit, translation in zip(units, translations):
            unit.target = translation
            unit.markfuzzy(False)

    @staticmethod
    def save(po_file: po.pofile, path: str) -> None:
        with open(path, "wb") as f:
            f.write(bytes(po_file))


class XliffTranslator:
    """Handles XLIFF file parsing, filtering, and output."""

    _XLIFF_NS = "{urn:oasis:names:tc:xliff:document:1.2}"

    @staticmethod
    def parse(path: str) -> xliff.xlifffile:
        return xliff.xlifffile.parsefile(path)

    @staticmethod
    def _get_state(unit: xliff.xliffunit) -> str:
        target_elem = unit.xmlelement.find(
            f"{XliffTranslator._XLIFF_NS}target"
        )
        if target_elem is not None:
            return target_elem.get("state", "")
        return ""

    @staticmethod
    def _get_translate_flag(unit: xliff.xliffunit) -> bool:
        return unit.xmlelement.get("translate", "yes").lower() != "no"

    @classmethod
    def get_untranslated(cls, xlf: xliff.xlifffile) -> list[xliff.xliffunit]:
        """Return units that need translation.

        Ported from translateXliffFile in src/translate.ts:394-404.
        """
        result: list[xliff.xliffunit] = []
        for unit in xlf.units:
            if not cls._get_translate_flag(unit):
                continue
            state = cls._get_state(unit).lower()
            target = (unit.target or "").strip()
            source = (unit.source or "").strip()

            state_needs = state.startswith("needs-") or state in ("new", "")
            has_meaningful = (
                bool(target)
                and target != source
                and not state_needs
            )
            if not has_meaningful:
                result.append(unit)
        return result

    @staticmethod
    def apply_batch(
        xlf: xliff.xlifffile,
        units: list[xliff.xliffunit],
        translations: list[str],
    ) -> None:
        for unit, translation in zip(units, translations):
            unit.target = translation
            # Update state on the target element
            target_elem = unit.xmlelement.find(
                f"{XliffTranslator._XLIFF_NS}target"
            )
            if target_elem is not None:
                target_elem.set("state", "translated")
            elif translation:
                # Target element didn't exist (was empty); create one
                import lxml.etree as ET

                new_target = ET.SubElement(
                    unit.xmlelement,
                    f"{XliffTranslator._XLIFF_NS}target",
                    {"state": "translated"},
                )
                new_target.text = translation

    @staticmethod
    def save(xlf: xliff.xlifffile, path: str) -> None:
        with open(path, "wb") as f:
            f.write(bytes(xlf))


def _run_translation(
    store,  # pofile | xlifffile
    units: list,  # list[pounit] | list[xliffunit]
    source_lang: str,
    target_lang: str,
    model: str,
    translator,  # PoTranslator | XliffTranslator
    output_path: str,
    context_file: str | None,
    context_length: int,
    timeout: int,
    verbose: bool,
) -> None:
    """Shared batch loop that orchestrates multi-turn translation."""
    if not verbose:
        litellm.set_verbose = False
        litellm.suppress_debug_info = True
    else:
        litellm.set_verbose = True
        litellm.suppress_debug_info = False

    system_prompt = load_system_prompt()
    user_prompt = load_user_prompt()
    context = _read_context(context_file)

    sources = [u.source for u in units]
    dict_entries = find_matching_entries(sources, target_lang)

    session = TranslationSession(model, timeout)
    session.setup(
        system_prompt, user_prompt, source_lang, target_lang,
        context, dict_entries,
    )

    console = Console()
    progress = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    )
    task_id = progress.add_task("", total=len(units))
    global_done = 0
    last_line = ""

    def on_unit_done(src: str, translation: str) -> None:
        nonlocal global_done, last_line
        global_done += 1
        progress.update(task_id, completed=global_done)
        if verbose:
            src_short = src[:70] + ("…" if len(src) > 70 else "")
            tgt_short = translation[:60] + ("…" if len(translation) > 60 else "")
            last_line = f"  {src_short} → {tgt_short}"
            progress.columns[-1].text = (
                f"{global_done}/"
                f"{progress.tasks[task_id].total}  {last_line}"
            )

    batch: list = []
    char_count = 0
    i = 0
    err429 = False

    with progress:
        while i < len(units):
            if err429:
                time.sleep(20)
                err429 = False

            unit = units[i]
            src_len = len(unit.source)
            if char_count < context_length:
                batch.append(unit)
                char_count += src_len
            if char_count >= context_length or i == len(units) - 1:
                try:
                    translations = session.translate_batch(
                        batch, on_progress=on_unit_done,
                    )

                    # on_progress may have already covered some; ensure all are applied
                    translator.apply_batch(store, batch, translations)
                    translator.save(store, output_path)

                    batch.clear()
                    char_count = 0
                except RateLimitError:
                    err429 = True
                    continue
                except Timeout:
                    console.print("\n[yellow]Timeout. Retrying...[/]")
                    continue
                except ParseError as e:
                    console.print(f"\n[red]Parse error: {e}. Retrying...[/]")
                    continue
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/]")
                    if getattr(e, "code", None) == "ECONNABORTED":
                        console.print(
                            '[yellow]You may need to set "HTTPS_PROXY" '
                            "to reach the API.[/]"
                        )
                    if len(session.messages) >= 2:
                        session.messages.pop()
                        session.messages.pop()
                    continue

            i += 1


def translate_po(
    model: str,
    po_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    output_path: str,
    context_file: str | None,
    context_length: int,
    timeout: int,
    fold_length: int = 120,
    sort_output: bool = False,
    escape_chars: bool = True,
) -> None:
    """Translate a single PO file."""
    translator = PoTranslator()
    po_file = translator.parse(po_path)

    # Determine target language
    if not target_lang:
        target_lang = translator.get_header_language(po_file)
    if not target_lang:
        print("No target language specified via --lang or PO header", file=__import__("sys").stderr)
        return

    # Trim fuzzy targets in-place
    translator.trim_fuzzy_targets(po_file)

    # Collect untranslated units
    untranslated = translator.get_untranslated(po_file)
    if not untranslated:
        print("All entries already translated.")
        translator.save(po_file, output_path)
        return

    po_file.updateheader(**{"Last-Translator": "aitran v0.1.0"})

    # Deterministic sort for consistent batch content
    untranslated.sort(key=lambda u: u.source)

    _run_translation(
        store=po_file,
        units=untranslated,
        source_lang=source_lang,
        target_lang=target_lang,
        model=model,
        translator=translator,
        output_path=output_path,
        context_file=context_file,
        context_length=context_length,
        timeout=timeout,
        verbose=verbose,
    )
    print()


def translate_po_dir(
    model: str,
    dir_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    context_file: str | None,
    context_length: int,
    timeout: int,
    fold_length: int = 120,
    sort_output: bool = False,
    escape_chars: bool = True,
) -> None:
    """Translate all .po files in a directory."""
    import os

    for entry in sorted(os.listdir(dir_path)):
        if entry.endswith(".po"):
            po_path = os.path.join(dir_path, entry)
            print(f"Translating {po_path}")
            translate_po(
                model, po_path, source_lang, target_lang,
                verbose, po_path, context_file, context_length, timeout,
                fold_length, sort_output, escape_chars,
            )


def translate_xliff_file(
    model: str,
    xliff_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    output_path: str,
    context_file: str | None,
    context_length: int,
    timeout: int,
) -> None:
    """Translate a single XLIFF file."""
    translator = XliffTranslator()
    xlf = translator.parse(xliff_path)

    if not xlf.units:
        print("No translation units found.")
        return

    # Determine source/target locale
    src = source_lang or xlf.sourcelanguage or "en"
    tgt = target_lang
    if not tgt:
        tgt = xlf.targetlanguage
    if not tgt:
        print("No target language specified via --lang or XLIFF header", file=__import__("sys").stderr)
        return

    # Set locales on units that lack them (ported from src/translate.ts:387-392)
    import os

    basename = os.path.basename(xliff_path)
    for unit in xlf.units:
        if not unit.xmlelement.get("source-language"):
            pass  # source locale is on <file> element, handled by translate-toolkit
        # ensure each unit has data set
        if not getattr(unit, "_source_locale_set", False):
            unit.xmlelement.set("source-language", src)
            unit.xmlelement.set("target-language", tgt)
            unit._source_locale_set = True

    untranslated = translator.get_untranslated(xlf)
    if not untranslated:
        print("All translation units are already translated.")
        translator.save(xlf, output_path)
        return

    _run_translation(
        store=xlf,
        units=untranslated,
        source_lang=src,
        target_lang=tgt,
        model=model,
        translator=translator,
        output_path=output_path,
        context_file=context_file,
        context_length=context_length,
        timeout=timeout,
        verbose=verbose,
    )
    print()


def translate_xliff_dir(
    model: str,
    dir_path: str,
    source_lang: str,
    target_lang: str,
    verbose: bool,
    context_file: str | None,
    context_length: int,
    timeout: int,
) -> None:
    """Translate all .xliff/.xlf files in a directory."""
    import os

    for entry in sorted(os.listdir(dir_path)):
        if entry.endswith((".xliff", ".xlf")):
            xliff_path = os.path.join(dir_path, entry)
            print(f"Translating {xliff_path}")
            translate_xliff_file(
                model, xliff_path, source_lang, target_lang,
                verbose, xliff_path, context_file, context_length, timeout,
            )
