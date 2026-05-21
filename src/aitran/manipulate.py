"""Remove PO entries by filter criteria."""

import re

from translate.storage import po

_FLAG_MAP: dict[str, int] = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "u": re.UNICODE,
    "y": 0,   # sticky — no Python equivalent, ignore
    "g": 0,   # global — not a regex compile flag in Python, ignore
    "x": re.VERBOSE,
}


def _parse_reference_filter(
    reference_contains: str,
) -> re.Pattern | str | None:
    """Parse a reference filter string.

    If it matches /pattern/flags, compile a regex. Otherwise treat as plain
    substring match.
    """
    m = re.match(r"^/([^/]+)/([igmsuy]*)$", reference_contains)
    if m:
        flags = 0
        for c in m.group(2):
            flags |= _FLAG_MAP.get(c, 0)
        return re.compile(m.group(1), flags)
    return reference_contains


def remove_by_options(
    po_path: str,
    output: str,
    fuzzy: bool = False,
    obsolete: bool = False,
    untranslated: bool = False,
    translated: bool = False,
    translated_not_fuzzy: bool = False,
    fuzzy_translated: bool = False,
    reference_contains: str | None = None,
    compile_opts: dict | None = None,
) -> None:
    """Remove entries from a PO file matching the given filters."""
    po_file = po.pofile.parsefile(po_path)

    ref_filter = _parse_reference_filter(reference_contains) if reference_contains else None

    to_remove: list[po.pounit] = []
    for unit in po_file.units:
        if unit.isheader():
            continue

        target = unit.target
        target_empty = not target or (isinstance(target, str) and not target.strip())
        is_fuzzy = unit.isfuzzy()
        is_obs = unit.isobsolete() if hasattr(unit, "isobsolete") else False

        remove = False
        if fuzzy and is_fuzzy:
            remove = True
        elif obsolete and is_obs:
            remove = True
        elif untranslated and target_empty:
            remove = True
        elif translated and not target_empty:
            remove = True
        elif translated_not_fuzzy and not target_empty and not is_fuzzy:
            remove = True
        elif fuzzy_translated and not target_empty and is_fuzzy:
            remove = True
        elif ref_filter is not None:
            locations = unit.getlocations() if hasattr(unit, "getlocations") else []
            ref = "\n".join(locations)
            if isinstance(ref_filter, re.Pattern):
                if ref_filter.search(ref):
                    remove = True
            elif ref_filter in ref:
                remove = True

        if remove:
            to_remove.append(unit)

    for unit in to_remove:
        po_file.removeunit(unit)

    with open(output, "wb") as f:
        f.write(bytes(po_file))
