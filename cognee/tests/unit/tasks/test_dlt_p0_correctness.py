"""Unit tests for DLT ingestion P0 correctness fixes.

Covers the DB-free behaviors:
  - dropped FK references are recorded for reporting (no longer silent)
  - unparseable external_metadata is logged with the object id (no longer silent)

The DB-dependent behaviors (PK-collision warning, failed-orphan tracking, and
deferred orphan cleanup) require mocked engines and belong with the broader
mocked DLT suite.
"""

from types import SimpleNamespace

from cognee.tasks.ingestion.dlt_utils import parse_external_metadata
from cognee.tasks.ingestion.resolve_dlt_sources import _resolve_fk_references


def _row(table_name, foreign_keys, row_data):
    return SimpleNamespace(
        table_name=table_name,
        foreign_keys=foreign_keys,
        row_data=row_data,
    )


def test_resolved_fk_reference_is_returned_when_target_loaded():
    row = _row(
        "orders",
        [{"column": "cust_id", "ref_table": "customers", "ref_column": "id"}],
        {"cust_id": 7},
    )
    target_id = "11111111-1111-1111-1111-111111111111"
    fk_lookup = {("customers", "7"): target_id}

    missing = []
    refs = _resolve_fk_references(row, fk_lookup, missing)

    assert missing == []
    assert len(refs) == 1
    assert refs[0]["target_table"] == "customers"
    assert refs[0]["target_node_id"] == str(target_id)


def test_missing_fk_target_is_recorded_not_silently_dropped():
    row = _row(
        "orders",
        [{"column": "cust_id", "ref_table": "customers", "ref_column": "id"}],
        {"cust_id": 99},
    )

    missing = []
    refs = _resolve_fk_references(row, {}, missing)  # empty lookup -> target missing

    assert refs == []  # no edge created
    assert missing == [("orders", "cust_id", "customers", "99")]


def test_missing_targets_optional_arg_is_backwards_compatible():
    row = _row(
        "orders",
        [{"column": "cust_id", "ref_table": "customers", "ref_column": "id"}],
        {"cust_id": 99},
    )
    # Omitting the accumulator must not raise (default None).
    assert _resolve_fk_references(row, {}) == []


def test_null_fk_value_is_skipped_without_being_recorded():
    row = _row(
        "orders",
        [{"column": "cust_id", "ref_table": "customers", "ref_column": "id"}],
        {"cust_id": None},
    )
    missing = []
    assert _resolve_fk_references(row, {}, missing) == []
    # A NULL FK column is a legitimate absence, not a dropped edge.
    assert missing == []


def test_parse_external_metadata_dict_passthrough():
    obj = SimpleNamespace(external_metadata={"source": "dlt"}, id="d1")
    assert parse_external_metadata(obj) == {"source": "dlt"}


def test_parse_external_metadata_valid_json_string():
    obj = SimpleNamespace(external_metadata='{"source": "dlt"}', id="d2")
    assert parse_external_metadata(obj) == {"source": "dlt"}


def test_parse_external_metadata_absent_is_silent(caplog):
    obj = SimpleNamespace(external_metadata=None, id="d3")
    with caplog.at_level("WARNING"):
        assert parse_external_metadata(obj) is None
    assert "Failed to parse external_metadata" not in caplog.text


def test_parse_external_metadata_malformed_logs_with_id(caplog):
    obj = SimpleNamespace(external_metadata="{not valid json", id="doc-123")
    with caplog.at_level("WARNING"):
        assert parse_external_metadata(obj) is None
    assert "Failed to parse external_metadata" in caplog.text
    assert "doc-123" in caplog.text
