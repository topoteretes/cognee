"""Keys of a staged dlt source come from dlt's schema hints, not from reflection."""

from types import SimpleNamespace

from cognee.tasks.ingestion.ingest_dlt_source import (
    _apply_key_hints,
    _dlt_key_hints,
    _loaded_schema,
)


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
        "primary_key": ["id"],
        "foreign_keys": [{"column": "buyer", "ref_table": "shop.users", "ref_column": "code"}],
    }
    # A composite key keeps every column.
    assert schema["shop.line"] == {"primary_key": ["order_id", "n"], "foreign_keys": []}


def test_a_composite_key_resolves_to_all_its_columns():
    from cognee.tasks.ingestion.ingest_dlt_source import _key_value, _resolve_primary_key

    columns = ["order_id", "n", "qty"]
    assert _resolve_primary_key(None, {"primary_key": ["order_id", "n"]}, columns) == [
        "order_id",
        "n",
    ]
    assert _resolve_primary_key(None, {"primary_key": "order_id"}, columns) == ["order_id"]
    assert _resolve_primary_key(None, {"primary_key": None}, columns) == ["order_id"]
    assert _key_value({"order_id": 7, "n": 2, "qty": 5}, ["order_id", "n"]) == "7:2"


def test_a_table_without_a_hint_keeps_its_reflected_keys():
    reflected = {"primary_key": "id", "foreign_keys": [{"column": "a", "ref_table": "b"}]}
    schema = {"notes": dict(reflected)}

    _apply_key_hints(schema, key_hints={})

    assert schema["notes"] == reflected


def test_hints_come_from_the_schema_the_source_loaded_into():
    """A shared pipeline keeps the first schema it saw as the default; a source's
    tables live in the source's own schema, a bare resource's in the default."""
    from dlt.common.schema import Schema
    from dlt.extract.source import DltSource

    csv_schema, sql_schema = Schema("notes"), Schema("sql_database")
    pipeline = SimpleNamespace(
        default_schema=csv_schema,
        schemas={"notes": csv_schema, "sql_database": sql_schema},
    )
    sql_source = DltSource(Schema("sql_database"), section="shop")

    assert _loaded_schema(pipeline, sql_source) is sql_schema
    assert _loaded_schema(pipeline, SimpleNamespace(name="a bare resource")) is csv_schema
