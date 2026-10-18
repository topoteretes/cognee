"""A document row's ``cognee_node_set`` column, through the real dlt staging run (SDK-863).

The reserved column is declared as json at load time, so dlt stores the list on
the row instead of normalizing it into a child table, and rows that never set
it keep the content hash (and so the document id) they had before the column
existed.
"""

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


@pytest.fixture
def staging(tmp_path, monkeypatch):
    """The real dlt pipeline on a throwaway SQLite staging database."""
    dlt = pytest.importorskip("dlt")
    config = SimpleNamespace(
        db_provider="sqlite",
        db_path=str(tmp_path),
        db_host="",
        db_port=0,
        db_username="",
        db_password="",
    )
    monkeypatch.setattr(ingest, "get_relational_config", lambda: config)
    monkeypatch.setattr(
        ingest,
        "get_dlt_destination",
        lambda dlt_db_name: dlt.destinations.sqlalchemy(
            credentials={"database": str(tmp_path / dlt_db_name), "drivername": "sqlite"},
        ),
    )
    pipeline_factory = dlt.pipeline
    monkeypatch.setattr(
        dlt,
        "pipeline",
        lambda **kw: pipeline_factory(pipelines_dir=str(tmp_path / "pipelines"), **kw),
    )

    def source(name, rows, tag="notion"):
        @dlt.resource(name=name, primary_key="id")
        def documents():
            yield from rows

        resource = documents()
        setattr(resource, DOCUMENT_SOURCE_ATTR, tag)
        setattr(resource, PIPELINE_SCOPE_ATTR, name)
        return resource

    async def run(name, rows, tag="notion"):
        return await ingest.ingest_dlt_source(
            source(name, rows, tag),
            "brain",
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )

    return run


@pytest.mark.asyncio
async def test_list_column_stays_on_the_row_and_makes_no_child_table(staging):
    rows = await staging(
        "notion_pages",
        [{"id": "p1", "title": "T", "content": "C", NODE_SET_COLUMN: ["ws:root", "ws:child"]}],
    )

    # Without the json hint dlt would load a notion_pages__cognee_node_set child
    # table too, and document mode would read each of its rows as a document.
    assert set(rows.loaded_tables) == {"notion_pages"}
    assert [row.table_name for row in rows] == ["notion_pages"]
    item = _build_document_data_item(rows[0], None, "notion")
    assert item.node_set == ["notion:ws:root", "notion:ws:child"]
    assert item.data == "# T\n\nC"


@pytest.mark.asyncio
async def test_rows_that_never_set_the_column_keep_their_pre_contract_hash(staging):
    """A Drive-shaped row hashes as before the column existed, so its id is stable."""
    drive_row = {"id": "fileA", "title": "Q3 Plan", "content": "Ship it.", "url": "https://d/A"}

    rows = await staging("drive_folder_root", [drive_row], tag="google_drive")

    legacy_hash = hashlib.md5(
        json.dumps(drive_row, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert rows[0].content_hash == legacy_hash
    assert _build_document_data_item(rows[0], None, "google_drive").node_set is None


@pytest.mark.asyncio
async def test_a_bare_string_is_refused_not_dropped(staging):
    """dlt puts a non-list value in a variant column, so reading only the real one would
    ingest the row with no node sets. The sync stops and names the row instead."""
    from cognee.exceptions import CogneeValidationError

    rows = await staging(
        "notion_pages",
        [
            {"id": "p1", "title": "T", "content": "C", NODE_SET_COLUMN: "ws:root"},
            {"id": "p2", "title": "T", "content": "C", NODE_SET_COLUMN: ["ws:root"]},
        ],
    )
    by_id = {row.primary_key_value: row for row in rows}

    assert by_id["p1"].row_data[NODE_SET_COLUMN] is None
    assert by_id["p1"].row_data[f"{NODE_SET_COLUMN}__v_text"] == "ws:root"
    with pytest.raises(CogneeValidationError, match="row 'p1'.*cognee_node_set__v_text"):
        _build_document_data_item(by_id["p1"], None, "notion")
    assert _build_document_data_item(by_id["p2"], None, "notion").node_set == ["notion:ws:root"]
