"""The reserved node_set column must not move existing document ids (SDK-863).

``content_hash`` feeds the stable row id, so a connector that gains the
``cognee_node_set`` column (NULL on every existing row) has to hash exactly as
before; only a row that sets the column is re-hashed.
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


def test_other_null_columns_still_count():
    assert _row_content_hash({**ROW, "url": None}) == _legacy_hash({**ROW, "url": None})
    assert _row_content_hash({**ROW, "url": None}) != _legacy_hash(ROW)
