"""Shared test helpers."""

from translate.storage import po


def po_parse(content: str) -> po.pofile:
    """Parse a PO string into a pofile object."""
    return po.pofile.parsestring(content.encode())
