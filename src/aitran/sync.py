"""PO to POT sync — update PO from POT preserving existing translations."""

from translate.storage import po


def sync(
    po_path: str,
    pot_path: str,
    output_path: str,
) -> None:
    """Update a PO file from a POT file, keeping existing translations.

    For each (context, msgid) pair in the POT, if the PO has a translation,
    it is copied into the merged result.
    """
    po_file = po.pofile.parsefile(po_path)
    pot_file = po.pofile.parsefile(pot_path)

    for ctx, entries in pot_file.translations.items():
        for msgid in entries:
            if (
                po_file.translations.get(ctx, {}).get(msgid)
                and po_file.translations[ctx][msgid].msgstr[0]
            ):
                pot_file.translations[ctx][msgid] = po_file.translations[ctx][msgid]

    po_file.translations = pot_file.translations

    with open(output_path, "wb") as f:
        f.write(bytes(po_file))
