"""Unset columns must not move existing document ids (SDK-863, #6033).

``content_hash`` feeds the stable row id, so a connector that gains a column
— the reserved ``cognee_node_set`` column, or a nullable metadata column like
``thread_ts`` that dlt adds on a later export and reads back as NULL on every
existing row — has to hash exactly as before; only a row that sets the column
is re-hashed.
"""

import hashlib
import json

from cognee.tasks.ingestion.dlt_utils import NODE_SET_COLUMN
from cognee.tasks.ingestion.ingest_dlt_source import _row_content_hash

ROW = {"id": "p1", "title": "T", "content": "C"}


def _legacy_hash(row_dict: dict) -> str:
    return hashlib.md5(json.dumps(row_dict, sort_keys=True, default=str).encode()).hexdigest()


def test_rows_without_the_column_hash_as_before():
    assert _row_content_hash(ROW) == _legacy_hash(ROW)


def test_an_unset_reserved_column_does_not_change_the_hash():
    for empty in (None, [], "", "[]", "null"):
        assert _row_content_hash({**ROW, NODE_SET_COLUMN: empty}) == _legacy_hash(ROW), empty


def test_a_set_reserved_column_changes_the_hash_like_any_other_column():
    assert _row_content_hash({**ROW, NODE_SET_COLUMN: ["notion:ws:root"]}) != _legacy_hash(ROW)


def test_a_newly_appearing_nullable_column_does_not_change_the_hash():
    """#6033: the first thread reply makes dlt add ``thread_ts``; existing
    rows read back with NULL and must keep their old hash."""
    row_before = {**ROW, "user": "U1", "text": "Kickoff is on Monday."}
    row_after = {**row_before, "thread_ts": None, "user_id": None, "user_name": None}
    assert _row_content_hash(row_after) == _row_content_hash(row_before)
    assert _row_content_hash(row_after) == _legacy_hash(row_before)


def test_empty_forms_of_any_column_do_not_change_the_hash():
    for empty in (None, [], "", "[]", "null"):
        assert _row_content_hash({**ROW, "thread_ts": empty}) == _legacy_hash(ROW), empty


def test_a_set_metadata_column_changes_the_hash_like_any_other_column():
    assert _row_content_hash({**ROW, "thread_ts": "1717200000.000100"}) != _legacy_hash(ROW)
    assert _row_content_hash({**ROW, "url": "https://x"}) != _legacy_hash(ROW)


def test_a_column_going_from_set_to_unset_changes_the_hash():
    assert _row_content_hash({**ROW, "url": "https://x"}) != _row_content_hash({**ROW, "url": None})
