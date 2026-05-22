"""Tests for PO entry removal."""

import tempfile

from translate.storage import po

from aitran.manipulate import remove_by_options


def _make_po(entries: list[tuple[str, str, str]]) -> str:
    """Build a minimal PO file in a temp path.

    entries: list of (msgid, msgstr, flags) where flags may be 'fuzzy',
    'obsolete', or ''.
    """
    po_file = po.pofile()
    for msgid, msgstr, flags in entries:
        unit = po.pounit(msgid)
        if msgstr:
            unit.target = msgstr
        if "fuzzy" in flags:
            unit.markfuzzy()
        if "obsolete" in flags:
            unit.typecomments = "obsolete"
        po_file.addunit(unit)

    with tempfile.NamedTemporaryFile(suffix=".po", delete=False, mode="wb") as tmp:
        tmp.write(bytes(po_file))
    return tmp.name


def test_remove_fuzzy():
    path = _make_po([
        ("hello", "你好", "fuzzy"),
        ("world", "世界", ""),
    ])
    po_file = po.pofile.parsefile(path)
    # Verify fuzzy is marked
    fuzzy_units = [u for u in po_file.units if u.isfuzzy() and not u.isheader()]
    assert len(fuzzy_units) == 1

    remove_by_options(po_path=path, output=path, fuzzy=True)

    po_file = po.pofile.parsefile(path)
    sources = {u.source for u in po_file.units if not u.isheader()}
    assert "hello" not in sources
    assert "world" in sources


def test_remove_untranslated():
    path = _make_po([
        ("hello", "", ""),
        ("world", "世界", ""),
    ])
    remove_by_options(po_path=path, output=path, untranslated=True)

    po_file = po.pofile.parsefile(path)
    sources = {u.source for u in po_file.units if not u.isheader()}
    assert "hello" not in sources
    assert "world" in sources


def test_remove_translated():
    path = _make_po([
        ("hello", "你好", ""),
        ("world", "", ""),
    ])
    remove_by_options(po_path=path, output=path, translated=True)

    po_file = po.pofile.parsefile(path)
    sources = {u.source for u in po_file.units if not u.isheader()}
    assert "hello" not in sources
    assert "world" in sources


def test_remove_translated_not_fuzzy():
    path = _make_po([
        ("hello", "你好", "fuzzy"),
        ("world", "世界", ""),
        ("foo", "bar", ""),
    ])
    remove_by_options(po_path=path, output=path, translated_not_fuzzy=True)

    po_file = po.pofile.parsefile(path)
    sources = {u.source for u in po_file.units if not u.isheader()}
    # "world" and "foo" are translated not fuzzy → removed
    # "hello" is fuzzy → kept
    assert "hello" in sources
    assert "world" not in sources


def test_remove_fuzzy_translated():
    path = _make_po([
        ("hello", "你好", "fuzzy"),
        ("world", "世界", ""),
    ])
    remove_by_options(po_path=path, output=path, fuzzy_translated=True)

    po_file = po.pofile.parsefile(path)
    sources = {u.source for u in po_file.units if not u.isheader()}
    assert "hello" not in sources
    assert "world" in sources


def test_remove_reference_contains_plain_text():
    """Test plain text reference filtering."""
    po_file = po.pofile()
    u = po.pounit("hello")
    u.target = "你好"
    u.addlocation("src/app.py:42")
    po_file.addunit(u)

    with tempfile.NamedTemporaryFile(suffix=".po", delete=False, mode="wb") as tmp:
        tmp.write(bytes(po_file))

    remove_by_options(po_path=tmp.name, output=tmp.name, reference_contains="app.py")

    po_file = po.pofile.parsefile(tmp.name)
    sources = {u.source for u in po_file.units if not u.isheader()}
    assert "hello" not in sources
