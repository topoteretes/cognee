"""Keys of a staged dlt source come from dlt's schema hints, not from reflection."""

from types import SimpleNamespace

from cognee.tasks.ingestion.ingest_dlt_source import (
    _apply_key_hints,
    _dlt_key_hints,
    _loaded_schema,
    _source_tables,
)


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

    hints = _dlt_key_hints([orders, line])

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
        "users": {"primary_key": ["code"], "foreign_keys": []},
        "line": {"primary_key": ["order_id", "n"], "foreign_keys": []},
    }
    # Postgres reflects "<dataset>.<table>"; SQLite reflects bare names.
    schema = {
        "shop.orders": {"primary_key": None, "foreign_keys": []},
        "shop.users": {"primary_key": None, "foreign_keys": []},
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


def test_only_keys_to_unloaded_tables_are_dropped():
    # A composite key, a key to a unique non-primary column, and a key to a table
    # this run did not load. The table describes the first two; which rows they
    # find is decided at resolve time. The third has no table to point at.
    hints = _dlt_key_hints(
        [
            {
                "name": "shipment",
                "columns": {"id": {"primary_key": True}},
                "references": [
                    {
                        "referenced_table": "order_lines",
                        "columns": ["order_id", "line_no"],
                        "referenced_columns": ["order_id", "line_no"],
                    },
                    {
                        "referenced_table": "users",
                        "columns": ["employee_no"],
                        "referenced_columns": ["employee_no"],
                    },
                    {
                        "referenced_table": "carriers",
                        "columns": ["carrier_id"],
                        "referenced_columns": ["id"],
                    },
                ],
            },
        ]
    )
    schema = {
        "shipment": {"primary_key": None, "foreign_keys": []},
        "order_lines": {"primary_key": None, "foreign_keys": []},
        "users": {"primary_key": None, "foreign_keys": []},
    }

    _apply_key_hints(schema, hints)

    assert schema["shipment"]["foreign_keys"] == [
        {"column": "order_id", "ref_table": "order_lines", "ref_column": "order_id"},
        {"column": "line_no", "ref_table": "order_lines", "ref_column": "line_no"},
        {"column": "employee_no", "ref_table": "users", "ref_column": "employee_no"},
    ]


def test_a_reference_with_unaligned_column_lists_is_skipped():
    hints = _dlt_key_hints(
        [
            {
                "name": "orders",
                "columns": {},
                "references": [
                    {
                        "referenced_table": "users",
                        "columns": ["a", "b"],
                        "referenced_columns": ["id"],
                    },
                    {
                        "referenced_table": "shops",
                        "columns": ["shop"],
                        "referenced_columns": ["id"],
                    },
                ],
            }
        ]
    )

    assert hints["orders"]["foreign_keys"] == [
        {"column": "shop", "ref_table": "shops", "ref_column": "id"}
    ]


def test_hints_come_from_this_runs_resources_not_the_shared_schema():
    """The shared pipeline's schema keeps every reference an earlier source staged
    under a table name; only the resources say what this source declares. Names
    are normalized the way dlt names the staging tables."""
    import dlt
    from dlt.common.schema import Schema

    @dlt.source(name="sql_database")
    def shop():
        yield dlt.resource([{"Id": 1}], name="Users", primary_key="Id")
        yield dlt.resource([{"Id": 1, "UserId": 1}], name="Orders", primary_key="Id")

    stale = Schema("sql_database")
    stale.update_table(
        {
            "name": "orders",
            "columns": {"user_id": {"name": "user_id", "data_type": "bigint"}},
            "references": [
                {"referenced_table": "users", "columns": ["user_id"], "referenced_columns": ["id"]}
            ],
        }
    )
    pipeline = SimpleNamespace(default_schema=stale, schemas={"sql_database": stale})

    hints = _dlt_key_hints(_source_tables(pipeline, shop()))

    assert hints == {
        "users": {"primary_key": ["id"], "foreign_keys": []},
        "orders": {"primary_key": ["id"], "foreign_keys": []},
    }


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
