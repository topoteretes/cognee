"""Keys of a staged dlt source come from dlt's schema hints, not from reflection."""

from types import SimpleNamespace

from cognee.tasks.ingestion.ingest_dlt_source import _apply_key_hints, _dlt_key_hints


def _schema(*tables):
    return SimpleNamespace(data_tables=lambda: list(tables))


def test_merged_references_split_into_one_foreign_key_per_column():
    # dlt merges every foreign key to one table into a single entry.
    orders = {
        "name": "orders",
        "columns": {"id": {"primary_key": True}, "buyer": {}, "seller": {}},
        "references": [
            {
                "referenced_table": "users",
                "columns": ["seller", "buyer"],
                "referenced_columns": ["code", "code"],
            }
        ],
    }
    line = {
        "name": "line",
        "columns": {"order_id": {"primary_key": True}, "n": {"primary_key": True}},
    }

    hints = _dlt_key_hints(_schema(orders, line))

    assert hints["orders"] == {
        "primary_key": ["id"],
        "foreign_keys": [
            {"column": "seller", "ref_table": "users", "ref_column": "code"},
            {"column": "buyer", "ref_table": "users", "ref_column": "code"},
        ],
    }
    assert hints["line"] == {"primary_key": ["order_id", "n"], "foreign_keys": []}


def test_keys_follow_the_reflected_table_names():
    hints = {
        "orders": {
            "primary_key": ["id"],
            "foreign_keys": [{"column": "buyer", "ref_table": "users", "ref_column": "code"}],
        },
        "line": {"primary_key": ["order_id", "n"], "foreign_keys": []},
    }
    # Postgres reflects "<dataset>.<table>"; SQLite reflects bare names.
    schema = {
        "shop.orders": {"primary_key": None, "foreign_keys": []},
        "shop.line": {"primary_key": None, "foreign_keys": []},
    }

    _apply_key_hints(schema, hints)

    assert schema["shop.orders"] == {
        "primary_key": "id",
        "foreign_keys": [{"column": "buyer", "ref_table": "shop.users", "ref_column": "code"}],
    }
    # A composite key stays unset: rows are keyed on a single column.
    assert schema["shop.line"] == {"primary_key": None, "foreign_keys": []}
