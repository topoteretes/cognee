"""A document row's ``cognee_structure`` column, through the real dlt staging run (SDK-986).

The column says where a row sits in its source's tree. It is loaded as json so dlt
keeps the object on the row instead of flattening it into columns, it reaches
``system_metadata["structure"]``, and it is never part of the content hash, so moving a
row changes its structure but not its document id.
"""

import hashlib
import json

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.tasks.ingestion.dlt_utils import NODE_SET_COLUMN, STRUCTURE_COLUMN
from cognee.tasks.ingestion.ingest_dlt_source import _row_content_hash
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item

ROW = {"id": "p1", "title": "T", "content": "C"}
PAGE = {"kind": "page", "id": "parent", "document": True}
DATABASE = {"kind": "database", "id": "db", "name": "Tasks"}


def _legacy_hash(row_dict: dict) -> str:
    return hashlib.md5(json.dumps(row_dict, sort_keys=True, default=str).encode()).hexdigest()


def test_the_structure_column_never_changes_the_hash():
    for structure in (None, {"ancestors": []}, {"ancestors": [PAGE]}, {"ancestors": [DATABASE]}):
        assert _row_content_hash({**ROW, STRUCTURE_COLUMN: structure}) == _legacy_hash(ROW)


def test_the_node_set_column_still_changes_the_hash_when_set():
    row = {**ROW, STRUCTURE_COLUMN: {"ancestors": [PAGE]}, NODE_SET_COLUMN: ["notion:ws:root"]}
    assert _row_content_hash(row) != _legacy_hash(ROW)
    assert _row_content_hash(row) == _legacy_hash({**ROW, NODE_SET_COLUMN: ["notion:ws:root"]})


@pytest.mark.asyncio
async def test_object_column_stays_on_the_row_and_reaches_system_metadata(staging):
    structure = {"ancestors": [DATABASE, PAGE]}
    rows = await staging("notion_pages", [{**ROW, STRUCTURE_COLUMN: structure}])

    # Without the json hint dlt would flatten the object into cognee_structure__ancestors
    # (a child table for the list) and the row would come back without its structure.
    assert set(rows.loaded_tables) == {"notion_pages"}
    assert not [column for column in rows[0].row_data if column.startswith(f"{STRUCTURE_COLUMN}__")]
    item = _build_document_data_item(rows[0], None, "notion")
    assert item.system_metadata["structure"] == structure
    assert item.system_metadata["external_id"] == "p1"
    assert item.data == "# T\n\nC"


@pytest.mark.asyncio
async def test_a_row_at_the_top_of_its_tree_is_not_a_row_without_structure(staging):
    rows = await staging(
        "notion_pages",
        [
            {**ROW, STRUCTURE_COLUMN: {"ancestors": []}},
            {**ROW, "id": "p2"},
        ],
    )
    by_id = {row.primary_key_value: row for row in rows}

    top = _build_document_data_item(by_id["p1"], None, "notion")
    assert top.system_metadata["structure"] == {"ancestors": []}
    assert "structure" not in _build_document_data_item(by_id["p2"], None, "notion").system_metadata


@pytest.mark.asyncio
async def test_moving_a_row_keeps_its_hash_and_replaces_its_structure(staging):
    before = await staging("notion_pages", [{**ROW, STRUCTURE_COLUMN: {"ancestors": [PAGE]}}])
    after = await staging(
        "notion_pages",
        [{**ROW, STRUCTURE_COLUMN: {"ancestors": [{**PAGE, "id": "elsewhere"}]}}],
    )

    assert before[0].content_hash == after[0].content_hash == _legacy_hash(ROW)
    moved = _build_document_data_item(after[0], None, "notion")
    assert moved.system_metadata["structure"]["ancestors"][0]["id"] == "elsewhere"


@pytest.mark.asyncio
async def test_rows_that_never_set_the_column_keep_their_hash_and_metadata(staging):
    """Drive and Gmail rows are untouched: same hash, no structure key."""
    drive_row = {"id": "fileA", "title": "Q3 Plan", "content": "Ship it.", "url": "https://d/A"}

    rows = await staging("drive_folder_root", [drive_row], tag="google_drive")

    assert rows[0].content_hash == _legacy_hash(drive_row)
    metadata = _build_document_data_item(rows[0], None, "google_drive").system_metadata
    assert "structure" not in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("ws:root", id="bare string"),
        pytest.param({"parent": "x"}, id="no ancestors"),
        pytest.param({"ancestors": "x"}, id="ancestors not a list"),
        pytest.param({"ancestors": [{"kind": "page"}]}, id="entry without an id"),
        pytest.param({"ancestors": [{"kind": "page", "id": "a", "document": "yes"}]}, id="flag"),
        pytest.param({"ancestors": ["page"]}, id="entry not an object"),
    ],
)
async def test_a_malformed_structure_is_refused_not_dropped(staging, bad):
    rows = await staging("notion_pages", [{**ROW, STRUCTURE_COLUMN: bad}])

    with pytest.raises(CogneeValidationError, match="row 'p1'.*cognee_structure"):
        _build_document_data_item(rows[0], None, "notion")
