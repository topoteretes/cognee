"""A NUL character in one document row must not reach the staging database, for any connector.

PostgreSQL rejects NUL in text, and one such row fails the load of the whole dlt package and of
every retry of it. Document sources are staged through ``ingest_dlt_source``, so it removes NUL
there once, with the rule the graph and vector adapters already use
(``sanitize_relational_payload``), instead of each connector carrying its own. These tests
run the real pipeline on SQLite, which stores NUL: what they check is the staged text.
"""

import importlib

import pytest

from cognee.modules.graph.methods.sanitize_relational_payload import sanitize_relational_payload
from cognee.tasks.ingestion.connectors import obsidian_source

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")

# Every connector under cognee/tasks/ingestion/connectors/ tags its resource with one of these.
DOCUMENT_SOURCES = [
    "apollo",
    "asana",
    "gmail",
    "google_calendar",
    "google_drive",
    "linear",
    "notion",
    "obsidian",
    "onenote",
]


def _document(row_id, title="Title", content="Body"):
    return {"id": row_id, "title": title, "content": content, "url": None, "_deleted": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", DOCUMENT_SOURCES)
async def test_document_rows_are_staged_without_nul_and_their_neighbours_load(staging, tag):
    text = "pre\x00post\nCafé 東京\t"
    rows = await staging(
        "folder",
        [
            _document("clean"),
            _document("poisoned", title="Ti\x00tle", content=text),
            _document("only_nul", content="\x00"),
        ],
        tag=tag,
    )

    staged = {row.primary_key_value: row.row_data for row in rows}
    assert set(staged) == {"clean", "poisoned", "only_nul"}
    assert staged["clean"]["content"] == "Body"
    # The one rule for NUL in the project: removed, not replaced.
    assert (
        staged["poisoned"]["content"] == sanitize_relational_payload(text) == "prepost\nCafé 東京\t"
    )
    assert staged["poisoned"]["title"] == "Title"
    assert staged["only_nul"]["content"] == ""
    assert not any("\x00" in str(value) for row in staged.values() for value in row.values())


@pytest.mark.asyncio
async def test_obsidian_note_with_nul_is_staged_without_it(staging, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note.md").write_bytes(b"alpha\x00beta\n")
    (vault / "Other.md").write_bytes(b"clean note\n")

    rows = await ingest.ingest_dlt_source(
        obsidian_source(vault_path=vault),
        "brain",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    staged = {row.primary_key_value: row.row_data["content"] for row in rows}
    assert staged == {"obsidian:Note.md": "alphabeta\n", "obsidian:Other.md": "clean note\n"}


def test_a_source_is_sanitized_once_and_only_text_changes():
    dlt = pytest.importorskip("dlt")
    row = {
        "id": "a",
        "content": "x\x00y",
        "_deleted": True,
        "url": None,
        "node_set": ["n\x00s"],
        "structure": {"ancestors": [{"kind": "page", "id": "p\x001"}]},
    }

    @dlt.resource(name="documents")
    def documents():
        yield row

    resource = documents()
    ingest.strip_nul_from_rows(resource)
    ingest.strip_nul_from_rows(resource)  # staging the same source again must not stack the step

    assert list(resource) == [
        {
            "id": "a",
            "content": "xy",
            "_deleted": True,
            "url": None,
            "node_set": ["ns"],
            "structure": {"ancestors": [{"kind": "page", "id": "p1"}]},
        }
    ]
