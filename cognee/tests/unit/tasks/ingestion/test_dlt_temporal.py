"""DLT rows and time: ISO-shaped cells become edges into shared Timestamp nodes.

``dlt_temporal`` decides which cells are points in time (by the value's shape,
since dlt types a bare date as TEXT), ``resolve_dlt_sources`` records them on
the manifest row, and ``emit_dlt_schema_graph`` writes one column-named edge
per cell into the ``Timestamp`` node ``timestamp_from_text`` builds — the node
every other mention of that instant resolves to.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import cognee.tasks.ingestion.dlt_schema_graph as schema_graph_module
from cognee.modules.engine.models import Timestamp
from cognee.modules.engine.utils.timestamp_from_text import timestamp_from_text
from cognee.tasks.ingestion.dlt_row_data import DltRowData
from cognee.tasks.ingestion.dlt_schema_graph import emit_dlt_schema_graph
from cognee.tasks.ingestion.dlt_temporal import temporal_cells, timestamp_str_for_cell

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2024-03-01", "2024-03-01"),
        ("2024-03-02 10:15:00", "2024-03-02 10:15:00"),
        ("2024-03-02 10:15:00.000000", "2024-03-02 10:15:00"),  # dlt's sqlite DATETIME
        ("2024-03-02T10:15:00+00:00", "2024-03-02 10:15:00"),  # Postgres timestamptz as str
        ("2024-03-02T10:15:00.250Z", "2024-03-02 10:15:00"),
        (" 2024-03-01 ", "2024-03-01"),
        ("2024-03", "2024-03"),  # monthly series (Datahub gold prices)
    ],
)
def test_iso_shaped_cells_normalize_to_the_forms_timestamp_from_text_accepts(value, expected):
    assert timestamp_str_for_cell(value) == expected
    assert timestamp_from_text(expected) is not None


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "1947",
        1947,
        "March 2024",
        "1 March 2024",
        "10:15:00",
        "2024-03 10:15:00",
        "ORD-2024-03-01",
    ],
)
def test_anything_but_an_iso_date_or_datetime_is_not_a_time_cell(value):
    assert timestamp_str_for_cell(value) is None


def _row(row_data, foreign_keys=None, primary_key_column="id"):
    return DltRowData(
        table_name="orders",
        primary_key_column=primary_key_column,
        primary_key_value="1",
        row_data=row_data,
        content_hash="hash",
        schema_info=[],
        schema_hash="schema-hash",
        foreign_keys=foreign_keys or [],
        dlt_db_name="shop",
        dataset_name="ds",
    )


def test_temporal_cells_skip_foreign_keys_and_non_dates():
    row = _row(
        {
            "id": 7,
            "customer_id": "2024-03-02",  # foreign keys are edges already
            "order_date": "2024-03-01",
            "shipped_at": "2024-03-02 10:15:00.000000",
            "note": "shipped 2024-03-02",
            "amount": 120.5,
            "returned_at": None,
        },
        foreign_keys=[{"column": "customer_id", "ref_table": "customers", "ref_column": "id"}],
    )
    assert temporal_cells(row) == {
        "order_date": "2024-03-01",
        "shipped_at": "2024-03-02 10:15:00",
    }


def test_a_date_primary_key_is_the_time_the_row_is_about():
    """A time series has no id: dlt keys it by its first column, the date."""
    row = _row({"date": "2009-03-01", "sp500": 757.13}, primary_key_column="date")
    assert temporal_cells(row) == {"date": "2009-03-01"}
    monthly = _row({"date": "2020-01", "price": 1561.0}, primary_key_column="date")
    assert temporal_cells(monthly) == {"date": "2020-01"}
    assert timestamp_from_text("2020-01").precision == "month"


def _stub_graph(monkeypatch):
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(schema_graph_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": None}),
    )
    return graph


@pytest.mark.asyncio
async def test_emit_links_rows_to_the_shared_timestamp_node_per_column(monkeypatch):
    graph = _stub_graph(monkeypatch)
    row_a, row_b = str(uuid4()), str(uuid4())
    row_records = [
        {
            "source_id": row_a,
            "table_name": "orders",
            "fk_references": [],
            "timestamps": {"order_date": "2024-03-01", "shipped_at": "2024-03-02 10:15:00"},
        },
        {
            "source_id": row_b,
            "table_name": "orders",
            "fk_references": [],
            "timestamps": {"order_date": "2024-03-01", "shipped_at": "2024-02-30 08:00:00"},
        },
    ]

    await emit_dlt_schema_graph({}, row_records, ctx=None)

    added_nodes = graph.add_nodes.call_args.args[0]
    timestamps = {node.timestamp_str: node for node in added_nodes if isinstance(node, Timestamp)}
    # One node per instant, shared by both rows; the impossible date is skipped.
    assert set(timestamps) == {"2024-03-01", "2024-03-02 10:15:00"}
    assert timestamps["2024-03-01"].precision == "day"
    assert timestamps["2024-03-01"].id == timestamp_from_text("2024-03-01").id, (
        "a row's date must resolve to the node an LLM-extracted '1 March 2024' resolves to"
    )
    assert timestamps["2024-03-01"].id == timestamp_from_text("1 March 2024").id

    added_edges = graph.add_edges.call_args.args[0]
    date_edges = [edge for edge in added_edges if edge[2] == "order_date"]
    assert {edge[0] for edge in date_edges} == {UUID(row_a), UUID(row_b)}
    assert {edge[1] for edge in date_edges} == {timestamps["2024-03-01"].id}
    assert all(edge[3]["edge_text"] == "order_date 2024-03-01" for edge in date_edges)
    shipped_edges = [edge for edge in added_edges if edge[2] == "shipped_at"]
    assert [edge[0] for edge in shipped_edges] == [UUID(row_a)]


@pytest.mark.asyncio
async def test_emit_without_timestamps_adds_no_timestamp_nodes(monkeypatch):
    graph = _stub_graph(monkeypatch)
    await emit_dlt_schema_graph(
        {}, [{"source_id": str(uuid4()), "table_name": "orders", "fk_references": []}], ctx=None
    )
    added = graph.add_nodes.call_args.args[0] if graph.add_nodes.call_args else []
    assert not any(isinstance(node, Timestamp) for node in added)
