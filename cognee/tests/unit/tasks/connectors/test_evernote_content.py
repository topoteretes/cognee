# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""ENML to text, and the document row a note becomes."""

import pytest

from cognee.tasks.ingestion.connectors.evernote import enml_to_text, note_to_row

pytestmark = pytest.mark.timeout(60)


def test_enml_headings():
    enml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd">
<en-note>
<h1>Top Level Title</h1>
<h2>Section Title</h2>
<h3>Subsection Title</h3>
<h4>Fourth Level</h4>
</en-note>"""
    text = enml_to_text(enml)
    assert "# Top Level Title" in text
    assert "## Section Title" in text
    assert "### Subsection Title" in text
    assert "Fourth Level" in text
    # h4 should not have a markdown hash prefix
    assert "#### " not in text


def test_enml_lists():
    enml = """<en-note>
<ul>
  <li>First item</li>
  <li>Second item</li>
</ul>
</en-note>"""
    text = enml_to_text(enml)
    assert "- First item" in text
    assert "- Second item" in text


def test_enml_todos_checked_and_unchecked():
    enml = """<en-note>
<div><en-todo checked="true"/>Completed task</div>
<div><en-todo checked="false"/>Incomplete task</div>
<div><en-todo/>Default task</div>
</en-note>"""
    text = enml_to_text(enml)
    assert "[x] Completed task" in text
    assert "[ ] Incomplete task" in text
    assert "[ ] Default task" in text


def test_enml_entities():
    enml = "<en-note><div>One&nbsp;&amp;&nbsp;Two &lt;Three&gt; &quot;Quotes&quot;</div></en-note>"
    text = enml_to_text(enml)
    assert "&" in text
    assert "<Three>" in text
    assert '"Quotes"' in text
    assert "One" in text and "Two" in text


def test_enml_nested_divs():
    enml = """<en-note>
<div>
  <div>Outer Line</div>
  <div>
    <div>Inner Line A</div>
    <div>Inner Line B</div>
  </div>
</div>
</en-note>"""
    text = enml_to_text(enml)
    assert "Outer Line" in text
    assert "Inner Line A" in text
    assert "Inner Line B" in text
    # Runs of blank lines should be collapsed to at most one
    assert "\n\n\n" not in text


def test_enml_media_attachment():
    enml = """<en-note>
<div>Here is an attachment:</div>
<en-media type="image/png" hash="d41d8cd98f00b204e9800998ecf8427e"/>
</en-note>"""
    text = enml_to_text(enml)
    assert "[attachment: image/png]" in text
    # Binary/hash should not be inlined as content
    assert "d41d8cd98f00b204e9800998ecf8427e" not in text


def test_enml_crypt_is_redacted():
    enml = """<en-note>
<div>Public text</div>
<en-crypt hint="password is pet name">super secret cipher data 12345</en-crypt>
<div>More public text</div>
</en-note>"""
    text = enml_to_text(enml)
    assert "[encrypted content]" in text
    assert "super secret cipher data 12345" not in text
    assert "Public text" in text
    assert "More public text" in text


def test_enml_empty_input():
    assert enml_to_text("") == ""
    assert enml_to_text("   \n\t  ") == ""
    assert enml_to_text(None) == ""


def test_enml_malformed_markup_never_raises():
    malformed = "<<<not valid xml at all <div>open tag without close <b>bold"
    text = enml_to_text(malformed)
    assert isinstance(text, str)
    assert "open tag without close" in text


def test_enml_malformed_unclosed_comment_does_not_raise():
    text = enml_to_text("<!-- unclosed comment <div>Visible</div>")
    assert isinstance(text, str)


def test_enml_malformed_unclosed_cdata_does_not_raise():
    text = enml_to_text("<![CDATA[ unclosed cdata <div>Visible</div>")
    assert isinstance(text, str)


def test_enml_malformed_bogus_declaration_does_not_raise():
    text = enml_to_text("<!BOGUS DECLARATION><div>Visible</div>")
    assert isinstance(text, str)
    assert "Visible" in text


def test_enml_malformed_bad_entity_does_not_raise():
    text = enml_to_text("<en-note><div>&bad_entity; &amp; &unknown</div></en-note>")
    assert isinstance(text, str)
    assert "&" in text


def test_enml_unicode_text():
    enml = "<en-note><div>日本語のメモ 🚀 &amp; 한국어 🌟</div></en-note>"
    text = enml_to_text(enml)
    assert "日本語のメモ 🚀 & 한국어 🌟" in text


def test_enml_scripts_and_styles_ignored():
    enml = """<en-note>
<style>body { color: red; }</style>
<script>alert('evil');</script>
<div>Visible content</div>
</en-note>"""
    text = enml_to_text(enml)
    assert "Visible content" in text
    assert "color: red" not in text
    assert "evil" not in text


def test_enml_table_cells_of_a_row_are_separated():
    enml = (
        "<en-note><table><tr><td>Name</td><td>Price</td></tr>"
        "<tr><th>Milk</th><td>2</td></tr></table></en-note>"
    )

    assert enml_to_text(enml) == "Name | Price\nMilk | 2"


# --- Row building tests ---

NOTE = {
    "guid": "g1",
    "title": "Plan",
    "notebook": "nb1",
    "tags": ["t2", "t1"],
    "usn": 1,
    "live": True,
}


def test_note_to_row_holds_exactly_the_document_columns():
    row = note_to_row(NOTE, "Body", {"nb1": "Work"}, {"t1": "beta", "t2": "Alpha"})

    assert row == {
        "id": "note:g1",
        "title": "Plan",
        "content": "Notebook: Work\nTags: Alpha, beta\n\nBody",
        "_deleted": False,
    }


def test_note_to_row_leaves_out_unknown_notebooks_and_tags():
    row = note_to_row(NOTE, "Body", {}, {"t1": ""})

    assert row["content"] == "Body", "no header and no leading blank line"


def test_a_tag_only_edit_changes_the_content():
    before = note_to_row({**NOTE, "tags": []}, "Body", {"nb1": "Work"}, {"t1": "beta"})
    after = note_to_row({**NOTE, "tags": ["t1"]}, "Body", {"nb1": "Work"}, {"t1": "beta"})

    assert before["content"] != after["content"]
