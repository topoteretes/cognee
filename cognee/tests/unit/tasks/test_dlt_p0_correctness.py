"""Unit tests for DLT ingestion P0 correctness fixes.

Covers the DB-free behaviors:
  - dropped FK references are recorded for reporting (no longer silent)

The DB-dependent behaviors (PK-collision warning, failed-orphan tracking, and
deferred orphan cleanup) require mocked engines and belong with the broader
mocked DLT suite.
"""

from types import SimpleNamespace

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


def test_fk_to_a_column_the_target_is_not_keyed_on_creates_no_edge():
    # badges.employee_no references users.employee_no, but users rows are keyed on
    # id: looking the value up by id would link badge 100 to the wrong user.
    row = _row(
        "badges",
        [{"column": "employee_no", "ref_table": "users", "ref_column": "employee_no"}],
        {"employee_no": 2},
    )
    fk_lookup = {("users", "2"): "11111111-1111-1111-1111-111111111111"}
    missing, unkeyed = [], set()

    refs = _resolve_fk_references(row, fk_lookup, missing, {"users": ["id"]}, unkeyed)

    assert refs == []
    assert missing == []
    assert unkeyed == {("badges", "employee_no", "users", "employee_no", "id")}


def test_fk_to_one_column_of_a_composite_key_creates_no_edge():
    # order_lines rows are keyed on "order_id:line_no"; a key to order_id alone
    # cannot name one line, and its value could match a line of another order.
    row = _row(
        "shipments",
        [{"column": "order_id", "ref_table": "order_lines", "ref_column": "order_id"}],
        {"order_id": 1},
    )
    fk_lookup = {("order_lines", "1"): "11111111-1111-1111-1111-111111111111"}
    missing, unkeyed = [], set()

    refs = _resolve_fk_references(
        row, fk_lookup, missing, {"order_lines": ["order_id", "line_no"]}, unkeyed
    )

    assert refs == []
    assert missing == []
    assert unkeyed == {("shipments", "order_id", "order_lines", "order_id", "order_id, line_no")}
