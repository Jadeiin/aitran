"""PO/XLIFF document adapters used by translator and reviewer Agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from translate.misc import quote
from translate.misc.multistring import multistring
from translate.storage import po, xliff

from aitran.agents._base import safe_prompt_text
from aitran.utils import aitran_version

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aitran.agent_work import ReviewResult as ReviewedUnit
    from aitran.agent_work import TranslationResult as TranslatedUnit

_LEGACY_LANGUAGE_CODES = {
    "zh_Hans": "zh_CN",
    "zh_Hant": "zh_TW",
    "zh_Hans_SG": "zh_SG",
    "zh_Hant_HK": "zh_HK",
}


def _multistring(value) -> list[str]:
    """Extract all forms from a translate-toolkit string value.

    Returns:
        A list of strings — one per plural form for multistrings, or a
        single-element list for plain strings.
    """
    if hasattr(value, "strings"):
        return [str(s) for s in value.strings]
    return [str(value)]


_XML_ENTITY_CODEPOINTS = {
    "amp": ord("&"),
    "lt": ord("<"),
    "gt": ord(">"),
}


def _decode_serialized_markup(source: str, target: str) -> str:
    """Reverse prompt XML escaping only for entities introduced by source text.

    Returns:
        Target text with prompt serialization entities decoded only when needed.
    """
    entity_codepoints = {
        name: codepoint
        for name, codepoint in _XML_ENTITY_CODEPOINTS.items()
        if chr(codepoint) in source
    }
    if not entity_codepoints:
        return target
    return quote.entitydecode(target, entity_codepoints)


def _read_context(context_file: str | None) -> str:
    if not context_file:
        return ""
    with open(context_file, encoding="utf-8") as f:
        return f.read().strip()


def _last_translator() -> str:
    return f"aitran v{aitran_version()}"


class PoTranslator:
    """Handles PO file parsing, filtering, and output."""

    @staticmethod
    def parse(path: str) -> po.pofile:
        """Parse a PO file from disk.

        Returns:
            Parsed PO file object.
        """
        return po.pofile.parsefile(path)

    @staticmethod
    def get_target_language(po_file: po.pofile) -> str | None:
        """Infer target language from PO metadata.

        Returns:
            Target language string, or None if no language can be inferred.
        """
        if target_language := po_file.gettargetlanguage():
            return target_language

        language_header = po_file.parseheader().get("Language", "").strip()
        return _LEGACY_LANGUAGE_CODES.get(language_header)

    @staticmethod
    def get_untranslated(po_file: po.pofile) -> list[po.pounit]:
        """Return units that need translation (empty target or fuzzy).

        Returns:
            List of untranslated or fuzzy PO units.
        """
        result: list[po.pounit] = []
        plural_tags = po_file.get_plural_tags()
        for unit in po_file.units:
            if unit.isheader():
                continue
            if unit.istranslated() and not unit.isfuzzy():
                # istranslated only checks singular form;
                # verify all plural forms are present and non-empty.
                if unit.hasplural() and len(plural_tags) > 1:
                    targets = _multistring(unit.target)
                    if len(targets) < len(plural_tags) or any(
                        not t.strip() for t in targets
                    ):
                        result.append(unit)
                        continue
                continue
            result.append(unit)
        return result

    @staticmethod
    def apply_batch(
        po_file: po.pofile,
        units: list[po.pounit],
        results: list[TranslatedUnit],
    ) -> None:
        """Apply a batch of agent results."""
        plural_tags = po_file.get_plural_tags()
        for unit, result in zip(units, results, strict=True):
            cleaned = [safe_prompt_text(t) for t in result.targets]
            if unit.hasplural():
                if len(cleaned) != len(plural_tags):
                    result.fuzzy = True
                target = po.pounit.sync_plural_count(
                    multistring(cleaned),
                    plural_tags,
                )
            else:
                target = cleaned[0]
            unit.target = target
            unit.markfuzzy(result.fuzzy)
            if result.note:
                unit.addnote(result.note, origin="translator")

    @staticmethod
    def apply_review_batch(
        po_file: po.pofile,
        units_by_index: Mapping[int, po.pounit],
        results: list[ReviewedUnit],
        *,
        auto_fix: bool = False,
    ) -> None:
        """Apply review results to PO units.

        Without *auto_fix*, only marks ``revise``/``reject`` entries as fuzzy
        with a review note.  With *auto_fix*, also writes the corrected target
        and clears the fuzzy marker.
        """
        for result in results:
            unit = units_by_index[result.index]
            if auto_fix and result.corrected is not None:
                corrected = _decode_serialized_markup(
                    str(unit.source),
                    safe_prompt_text(result.corrected),
                )
                if unit.hasplural():
                    existing = _multistring(unit.target)
                    forms = [corrected, *existing[1:]]
                    unit.target = po.pounit.sync_plural_count(
                        multistring(forms),
                        po_file.get_plural_tags(),
                    )
                else:
                    unit.target = corrected
                unit.markfuzzy(unit.hasplural())
            else:
                unit.markfuzzy(True)
            if result.note:
                unit.addnote(
                    f"(review) {result.verdict}: {result.note}",
                    origin="translator",
                )

    @staticmethod
    def save(po_file: po.pofile, path: str) -> None:
        """Serialize a PO file to disk.

        Args:
            po_file: The in-memory PO file.
            path: Destination file path.
        """
        with open(path, "wb") as f:
            f.write(bytes(po_file))


class XliffTranslator:
    """Handles XLIFF file parsing, filtering, and output."""

    _DONE_STATES: ClassVar[set[str]] = {"final", "signed-off", "translated"}

    @staticmethod
    def parse(path: str) -> xliff.xlifffile:
        """Parse an XLIFF file from disk.

        Returns:
            Parsed XLIFF file object.  Files containing gettext plural
            groups (``restype="x-gettext-plurals"``) are automatically
            re-parsed as ``PoXliffFile`` so plural units are exposed
            correctly, even when the ``datatype`` is not ``"po"``.
        """
        from translate.storage import poxliff

        xlf = xliff.xlifffile.parsefile(path)
        if isinstance(xlf, poxliff.PoXliffFile):
            return xlf
        # Detect gettext plural groups that the default parser missed.
        ns = xlf.namespace
        groups = xlf.document.getroot().iterdescendants(
            f"{{{ns}}}group" if ns else "group"
        )
        if any(g.get("restype") == "x-gettext-plurals" for g in groups):
            return poxliff.PoXliffFile.parsestring(bytes(xlf))
        return xlf

    @staticmethod
    def _get_state(unit: xliff.xliffunit) -> str:
        tag = unit.namespaced("target")
        target_elem = unit.xmlelement.find(tag)
        if target_elem is not None:
            return target_elem.get("state", "")
        return ""

    @staticmethod
    def _get_translate_flag(unit: xliff.xliffunit) -> bool:
        return unit.xmlelement.get("translate", "yes").lower() != "no"

    @classmethod
    def get_untranslated(cls, xlf: xliff.xlifffile) -> list[xliff.xliffunit]:
        """Return units that need translation."""
        result: list[xliff.xliffunit] = []
        plural_tags = xlf.get_plural_tags()
        for unit in xlf.units:
            if not cls._get_translate_flag(unit):
                continue
            state = cls._get_state(unit).lower()

            if state.startswith("needs-") or state == "new":
                result.append(unit)
                continue

            # Plural units: check all forms individually.
            if unit.hasplural() and len(plural_tags) > 1:
                targets = _multistring(unit.target or "")
                if len(targets) < len(plural_tags) or any(
                    not t.strip() for t in targets
                ):
                    result.append(unit)
                continue

            target = (unit.target or "").strip()
            source = (unit.source or "").strip()
            if not target:
                result.append(unit)
                continue
            if state in cls._DONE_STATES:
                continue
            if target == source:
                result.append(unit)
        return result

    @staticmethod
    def apply_batch(
        xlf: xliff.xlifffile,
        units: list[xliff.xliffunit],
        results: list[TranslatedUnit],
    ) -> None:
        """Apply translation results to XLIFF units."""
        plural_tags = xlf.get_plural_tags()
        for unit, result in zip(units, results, strict=True):
            cleaned = [safe_prompt_text(t) for t in result.targets]
            if unit.hasplural():
                if len(cleaned) != len(plural_tags):
                    result.fuzzy = True
                unit.settarget(multistring(cleaned))
            else:
                unit.settarget(cleaned[0])
            if result.fuzzy:
                unit.markreviewneeded()
                # PoXliffUnit.markreviewneeded() does not propagate to
                # child trans-units unlike marktranslated/markfuzzy.
                if unit.hasplural():
                    for child in unit.units:
                        child.markreviewneeded()
            else:
                unit.marktranslated()
            if result.note:
                unit.addnote(result.note, origin="translator")

    @staticmethod
    def apply_review_batch(
        _store,
        units_by_index: Mapping[int, xliff.xliffunit],
        results: list[ReviewedUnit],
        *,
        auto_fix: bool = False,
    ) -> None:
        """Apply review results to XLIFF units.

        Without *auto_fix*, only marks ``revise``/``reject`` entries as
        needs-review with a note.  With *auto_fix*, also writes the corrected
        target and marks translated.
        """
        for result in results:
            unit = units_by_index[result.index]
            if auto_fix and result.corrected is not None:
                corrected = _decode_serialized_markup(
                    str(unit.source),
                    safe_prompt_text(result.corrected),
                )
                if unit.hasplural():
                    existing = _multistring(unit.target)
                    unit.settarget(multistring([corrected, *existing[1:]]))
                    unit.markfuzzy(True)
                else:
                    unit.settarget(corrected)
                    unit.marktranslated()
            else:
                unit.markreviewneeded()
                if unit.hasplural():
                    for child in unit.units:
                        child.markreviewneeded()
            if result.note:
                unit.addnote(
                    f"(review) {result.verdict}: {result.note}",
                    origin="translator",
                )

    @staticmethod
    def save(xlf: xliff.xlifffile, path: str) -> None:
        """Serialize an XLIFF file to disk.

        Args:
            xlf: The in-memory XLIFF file.
            path: Destination file path.
        """
        with open(path, "wb") as f:
            f.write(bytes(xlf))


def _order_units(units: list, order: str) -> list:
    """Reorder translation units by the chosen strategy.

    ``"file"`` (default) preserves the original PO file order.  ``"source"``
    sorts alphabetically for dedup / cache-hit friendliness.  ``"reference"``
    groups by the first source-code location so strings from the same file
    are translated together.  ``"context"`` groups by ``msgctxt`` so strings
    sharing a disambiguation context stay adjacent.

    Returns:
        Reordered list of units.
    """
    if order == "file" or not units:
        return units

    indexed = list(enumerate(units))
    if order == "source":
        indexed.sort(key=lambda item: item[1].source)
    elif order == "reference":

        def _reference_key(item):
            u = item[1]
            locs = u.getlocations() if hasattr(u, "getlocations") else []
            return (locs[0] if locs else "", item[0])

        indexed.sort(key=_reference_key)
    elif order == "context":

        def _context_key(item):
            u = item[1]
            getctx = getattr(u, "getcontext", None)
            ctx = getctx() if callable(getctx) else ""
            return (ctx, item[0])

        indexed.sort(key=_context_key)

    return [u for _, u in indexed]
