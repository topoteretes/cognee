"""A document row's ``cognee_structure`` column, through the real dlt staging run (SDK-986).

Loaded as json, so dlt keeps the object on the row instead of flattening it, and never
hashed, so moving a row changes its structure but not its document id.
"""

import hashlib
import json

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.tasks.ingestion.dlt_utils import STRUCTURE_COLUMN
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item

ROW = {"id": "p1", "title": "T", "content": "C"}
PAGE = {"kind": "page", "id": "parent", "document": True}


def _legacy_hash(row_dict: dict) -> str:
    return hashlib.md5(json.dumps(row_dict, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.asyncio
async def test_moving_a_row_keeps_its_hash_and_replaces_its_structure(staging):
    before = await staging("notion_pages", [{**ROW, STRUCTURE_COLUMN: {"ancestors": [PAGE]}}])
    moved = {"ancestors": [{"kind": "database", "id": "db", "name": "Tasks"}, PAGE]}
    after = await staging("notion_pages", [{**ROW, STRUCTURE_COLUMN: moved}])

    # Without the json hint dlt would flatten the object into cognee_structure__* columns.
    assert set(after.loaded_tables) == {"notion_pages"}
    assert before[0].content_hash == after[0].content_hash == _legacy_hash(ROW)
    item = _build_document_data_item(after[0], None, "notion")
    assert item.system_metadata["structure"] == moved


@pytest.mark.asyncio
async def test_rows_that_never_set_the_column_keep_their_hash_and_metadata(staging):
    """Drive and Gmail rows are untouched: same hash, no structure key."""
    drive_row = {"id": "fileA", "title": "Q3 Plan", "content": "Ship it.", "url": "https://d/A"}

    rows = await staging("drive_folder_root", [drive_row], tag="google_drive")

    assert rows[0].content_hash == _legacy_hash(drive_row)
    assert (
        "structure" not in _build_document_data_item(rows[0], None, "google_drive").system_metadata
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        # dlt moves a scalar into a cognee_structure__v_text variant column.
        pytest.param("ws:root", id="bare string"),
        pytest.param({"ancestors": [{"kind": "page"}]}, id="entry without an id"),
        pytest.param(
            {"ancestors": [{"kind": "page", "id": "a", "color": "red"}]}, id="unknown key"
        ),
        pytest.param(
            {"ancestors": [PAGE, {"kind": "database", "id": "db"}]}, id="after a document"
        ),
        pytest.param({"ancestors": [{**PAGE, "id": "p1"}]}, id="the row itself"),
        pytest.param({"ancestors": [{"kind": "db", "id": "a", "name": 5}]}, id="name not text"),
        pytest.param({"ancestors": [{"kind": "db", "id": "a", "document": "yes"}]}, id="flag"),
        pytest.param({"parents": []}, id="no ancestors list"),
    ],
)
async def test_a_malformed_structure_is_refused_not_dropped(staging, bad):
    rows = await staging("notion_pages", [{**ROW, STRUCTURE_COLUMN: bad}])

    with pytest.raises(CogneeValidationError, match="row 'p1'.*cognee_structure"):
        _build_document_data_item(rows[0], None, "notion")


def test_a_row_without_an_id_has_no_external_id_to_hang_children_under():
    """A custom source may emit id 0 or "". Its document gets no external_id, so the
    structure pass could not find it, and keeping the structure would make that pass
    fail for the whole dataset."""
    from cognee.tasks.ingestion.dlt_row_data import DltRowData

    row = DltRowData(
        table_name="t",
        primary_key_column="id",
        primary_key_value="0",
        row_data={"id": 0, "title": "T", "content": "C", STRUCTURE_COLUMN: {"ancestors": []}},
        content_hash="h",
        schema_info=[],
        schema_hash="s",
        foreign_keys=[],
        dlt_db_name="d",
        dataset_name="brain",
    )

    metadata = _build_document_data_item(row, None, "custom").system_metadata

    assert "external_id" not in metadata and "structure" not in metadata
