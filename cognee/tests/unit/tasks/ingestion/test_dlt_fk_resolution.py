"""A row's foreign keys find their target rows: by the whole key the target is keyed on,
or, for a key to a UNIQUE column, by the one row holding the referenced value."""

from types import SimpleNamespace

from cognee.tasks.ingestion.resolve_dlt_sources import _column_index, _resolve_fk_references

LINE_12 = "11111111-1111-1111-1111-111111111111"
LINE_21 = "22222222-2222-2222-2222-222222222222"
ANN = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOB = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


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
    missing, ambiguous = [], set()

    refs = _resolve_fk_references(
        row, lookup, missing, {"order_lines": ["order_id", "line_no"]}, None, ambiguous
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
    assert missing == [] and ambiguous == set()


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

    refs = _resolve_fk_references(row, lookup, [], {"order_lines": ["order_id", "line_no"]})

    assert _edges(refs) == {
        ("o_l_references_order_lines", LINE_12),
        ("ro_rl_references_order_lines", LINE_21),
    }


def _lookup(rows_of):
    """column_lookup over target rows given as {table: [(row_data, node_id), ...]}."""
    return lambda table, columns: _column_index(
        [(SimpleNamespace(row_data=data), node_id) for data, node_id in rows_of.get(table, [])],
        columns,
    )


def test_a_key_to_a_unique_column_finds_the_row_holding_that_value():
    # users rows are keyed on id; looking employee_no 2 up by id would find Bob.
    row = _row("badges", [_fk("employee_no", "users", "employee_no")], {"employee_no": 2})
    users = [({"id": 1, "employee_no": 2}, ANN), ({"id": 2, "employee_no": 7}, BOB)]
    missing, ambiguous = [], set()

    refs = _resolve_fk_references(
        row,
        {("users", "1"): ANN, ("users", "2"): BOB},
        missing,
        {"users": ["id"]},
        _lookup({"users": users}),
        ambiguous,
    )

    assert refs == [
        {
            "column": "employee_no",
            "target_table": "users",
            "target_column": "employee_no",
            "target_pk_value": "2",
            "target_node_id": ANN,
            "relationship_name": "employee_no_references_users",
        }
    ]
    assert missing == [] and ambiguous == set()


def test_a_unique_value_several_rows_hold_creates_no_edge():
    row = _row("badges", [_fk("employee_no", "users", "employee_no")], {"employee_no": 2})
    users = [({"id": 1, "employee_no": 2}, ANN), ({"id": 2, "employee_no": 2}, BOB)]
    missing, ambiguous = [], set()

    refs = _resolve_fk_references(
        row, {}, missing, {"users": ["id"]}, _lookup({"users": users}), ambiguous
    )

    assert refs == [] and missing == []
    assert ambiguous == {("badges", "employee_no", "users", "employee_no")}


def test_a_composite_unique_key_finds_its_row_by_all_its_columns():
    # x alone is held by both rows; only (x, y) names one.
    row = _row("refs", [_fk("a", "pairs", "x"), _fk("b", "pairs", "y")], {"a": 1, "b": 2})
    pairs = [({"id": 10, "x": 1, "y": 1}, LINE_12), ({"id": 11, "x": 1, "y": 2}, LINE_21)]

    refs = _resolve_fk_references(row, {}, [], {"pairs": ["id"]}, _lookup({"pairs": pairs}), set())

    assert _edges(refs) == {("a_b_references_pairs", LINE_21)}


def test_a_composite_key_with_a_null_column_is_no_reference():
    row = _row(
        "shipments",
        [_fk("o", "order_lines", "order_id"), _fk("l", "order_lines", "line_no")],
        {"o": 1, "l": None},
    )
    missing, ambiguous = [], set()

    refs = _resolve_fk_references(
        row, {}, missing, {"order_lines": ["order_id", "line_no"]}, None, ambiguous
    )

    assert refs == [] and missing == [] and ambiguous == set()


def test_a_key_to_part_of_a_composite_key_needs_a_value_one_row_holds():
    # order_id alone is held by two order lines, so it names neither.
    row = _row(
        "shipments",
        [_fk("o", "order_lines", "order_id"), _fk("other", "users", "id")],
        {"o": 1, "other": 5},
    )
    lines = [({"order_id": 1, "line_no": 1}, LINE_12), ({"order_id": 1, "line_no": 2}, LINE_21)]
    ambiguous = set()

    refs = _resolve_fk_references(
        row,
        {("users", "5"): ANN},
        [],
        {"order_lines": ["order_id", "line_no"], "users": ["id"]},
        _lookup({"order_lines": lines}),
        ambiguous,
    )

    assert _edges(refs) == {("other_references_users", ANN)}
    assert ambiguous == {("shipments", "o", "order_lines", "order_id")}


def test_a_key_to_a_table_this_add_did_not_load_gets_its_own_warning(monkeypatch):
    # badges -> people found no row because no people were loaded by this add();
    # shipments -> order_lines missed one row of a table that was loaded.
    from importlib import import_module
    from unittest.mock import MagicMock

    # The package re-exports a function of the same name, so import the module itself.
    module = import_module("cognee.tasks.ingestion.resolve_dlt_sources")

    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    missing = [
        ("badges", "employee_no", "people", "2"),
        ("badges", "employee_no", "people", "7"),
        ("shipments", "order_id, line_no", "order_lines", "9:1"),
    ]

    module._report_missing_fk_targets(missing, {"badges", "shipments", "order_lines"})

    unloaded, unresolved = logger.warning.call_args_list
    assert unloaded.args[1:5] == ("badges", "employee_no", "people", 2)
    assert "only within one add()" in unloaded.args[0]
    assert unresolved.args[1:] == (1, "shipments.order_id, line_no -> order_lines:9:1")
    assert "max_rows_per_table" in unresolved.args[0]


def test_a_hint_missing_a_field_raises():
    """Hints are cognee's own output, so an incomplete one is a bug, not a key to skip."""
    import pytest

    row = _row("badges", [{"column": "owner_id", "ref_table": "people"}], {"owner_id": 1})
    with pytest.raises(KeyError):
        _resolve_fk_references(row, {}, [], {"people": ["id"]})

    row = _row("badges", [_fk("owner_id", "people", "")], {"owner_id": 1})
    with pytest.raises(ValueError):
        _resolve_fk_references(row, {}, [], {"people": ["id", "org"]})
