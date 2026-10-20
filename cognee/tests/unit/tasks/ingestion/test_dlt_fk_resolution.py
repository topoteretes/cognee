"""A row's foreign keys find their target rows by the whole key the target is keyed on."""

from types import SimpleNamespace

from cognee.tasks.ingestion.resolve_dlt_sources import _resolve_fk_references

LINE_12 = "11111111-1111-1111-1111-111111111111"
LINE_21 = "22222222-2222-2222-2222-222222222222"


def _row(table_name, foreign_keys, row_data):
    return SimpleNamespace(table_name=table_name, foreign_keys=foreign_keys, row_data=row_data)


def _fk(column, ref_table, ref_column):
    return {"column": column, "ref_table": ref_table, "ref_column": ref_column}


def _edges(refs):
    return {(ref["relationship_name"], ref["target_node_id"]) for ref in refs}


def test_a_composite_key_finds_its_row_by_all_its_columns():
    # Declared in a different column order than the target's key.
    row = _row(
        "shipments",
        [_fk("ship_line", "order_lines", "line_no"), _fk("ship_order", "order_lines", "order_id")],
        {"ship_order": 1, "ship_line": 2},
    )
    lookup = {("order_lines", "1:2"): LINE_12, ("order_lines", "2:1"): LINE_21}
    missing, unkeyed = [], set()

    refs = _resolve_fk_references(
        row, lookup, missing, {"order_lines": ["order_id", "line_no"]}, unkeyed
    )

    assert refs == [
        {
            "column": "ship_order, ship_line",
            "target_table": "order_lines",
            "target_column": "order_id, line_no",
            "target_pk_value": "1:2",
            "target_node_id": LINE_12,
            "relationship_name": "ship_order_ship_line_references_order_lines",
        }
    ]
    assert missing == [] and unkeyed == set()


def test_two_composite_keys_to_one_table_stay_apart():
    # dlt merges both keys to order_lines into one reference entry.
    row = _row(
        "returns",
        [
            _fk("o", "order_lines", "order_id"),
            _fk("l", "order_lines", "line_no"),
            _fk("ro", "order_lines", "order_id"),
            _fk("rl", "order_lines", "line_no"),
        ],
        {"o": 1, "l": 2, "ro": 2, "rl": 1},
    )
    lookup = {("order_lines", "1:2"): LINE_12, ("order_lines", "2:1"): LINE_21}

    refs = _resolve_fk_references(row, lookup, [], {"order_lines": ["order_id", "line_no"]}, set())

    assert _edges(refs) == {
        ("o_l_references_order_lines", LINE_12),
        ("ro_rl_references_order_lines", LINE_21),
    }


def test_a_key_to_a_column_the_target_is_not_keyed_on_creates_no_edge():
    # users rows are keyed on id: looking employee_no up by id would find the wrong user.
    row = _row("badges", [_fk("employee_no", "users", "employee_no")], {"employee_no": 2})
    unkeyed = set()

    refs = _resolve_fk_references(row, {("users", "2"): LINE_21}, [], {"users": ["id"]}, unkeyed)

    assert refs == []
    assert unkeyed == {("badges", "employee_no", "users", "employee_no", "id")}


def test_a_composite_key_with_a_null_column_is_no_reference():
    row = _row(
        "shipments",
        [_fk("o", "order_lines", "order_id"), _fk("l", "order_lines", "line_no")],
        {"o": 1, "l": None},
    )
    missing, unkeyed = [], set()

    refs = _resolve_fk_references(
        row, {}, missing, {"order_lines": ["order_id", "line_no"]}, unkeyed
    )

    assert refs == [] and missing == [] and unkeyed == set()


def test_an_incomplete_composite_key_is_reported_once():
    row = _row(
        "shipments",
        [_fk("o", "order_lines", "order_id"), _fk("other", "users", "id")],
        {"o": 1, "other": 5},
    )
    unkeyed = set()

    refs = _resolve_fk_references(
        row,
        {("users", "5"): LINE_21},
        [],
        {"order_lines": ["order_id", "line_no"], "users": ["id"]},
        unkeyed,
    )

    assert _edges(refs) == {("other_references_users", LINE_21)}
    assert unkeyed == {("shipments", "o", "order_lines", "order_id", "order_id, line_no")}
