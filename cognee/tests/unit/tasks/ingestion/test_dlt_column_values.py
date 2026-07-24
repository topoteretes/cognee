"""Tests for ColumnValue node emission in the DLT pipeline.

Column selection resolves per table (resolve_dlt_sources._resolve_column_value_selection:
explicit lists, "*" for all columns, "auto" cardinality gating) and per row
(_selected_column_values: pk/fk exclusion, value guards); emission
(emit_dlt_schema_graph) creates one shared ColumnValue node per unique
(table, column, value) with a column-named edge from each row.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.tasks.ingestion.dlt_schema_graph as schema_graph_module
from cognee.modules.engine.models import ColumnValue
from cognee.tasks.ingestion.dlt_row_data import DltRowData
from cognee.tasks.ingestion.dlt_schema_graph import emit_dlt_schema_graph
from cognee.tasks.ingestion.resolve_dlt_sources import (
    ALL_COLUMNS,
    MAX_COLUMN_VALUE_LENGTH,
    _resolve_column_value_selection,
    _selected_column_values,
)

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)


def _row(
    table_name="orders",
    row_data=None,
    foreign_keys=None,
    primary_key_column="id",
    primary_key_value="1",
):
    return DltRowData(
        table_name=table_name,
        primary_key_column=primary_key_column,
        primary_key_value=primary_key_value,
        row_data=row_data or {},
        content_hash=f"hash-{primary_key_value}",
        schema_info=[],
        schema_hash="schema-hash",
        foreign_keys=foreign_keys or [],
        dlt_db_name="shop",
        dataset_name="ds",
    )


def _unique_rows(rows):
    return {(row.table_name, row.primary_key_value, row.content_hash): row for row in rows}


class TestResolveColumnValueSelection:
    def test_disabled(self):
        rows = _unique_rows([_row(row_data={"id": 1, "status": "active"})])
        assert _resolve_column_value_selection(rows, {}) == {}
        assert _resolve_column_value_selection(rows, None) == {}

    def test_explicit_columns(self):
        rows = _unique_rows([_row(row_data={"id": 1, "status": "active"})])
        resolved = _resolve_column_value_selection(rows, {"orders": ["status"]})
        assert resolved == {"orders": {"status"}}

    def test_table_wildcard_applies_to_all_tables(self):
        rows = _unique_rows(
            [
                _row(table_name="orders", row_data={"id": 1}),
                _row(table_name="customers", row_data={"id": 2}),
            ]
        )
        resolved = _resolve_column_value_selection(rows, {"*": ["country"]})
        assert resolved == {"orders": {"country"}, "customers": {"country"}}

    def test_all_columns_bypasses_cardinality_gate(self):
        rows = _unique_rows(
            [
                _row(row_data={"id": i, "ts": f"2026-07-24T{i:02d}"}, primary_key_value=str(i))
                for i in range(4)
            ]
        )
        resolved = _resolve_column_value_selection(rows, {"orders": ["*"]})
        assert resolved == {"orders": {ALL_COLUMNS}}

    def test_auto_keeps_repeating_and_drops_unique_columns(self):
        # status repeats (2 distinct / 4 rows); ts and text are unique per row.
        rows = _unique_rows(
            [
                _row(
                    row_data={
                        "id": i,
                        "status": "active" if i % 2 else "closed",
                        "ts": f"2026-07-24T00:00:{i:02d}",
                        "text": f"unique message {i}",
                    },
                    primary_key_value=str(i),
                )
                for i in range(4)
            ]
        )
        resolved = _resolve_column_value_selection(rows, {"*": ["auto"]})
        assert resolved == {"orders": {"status"}}

    def test_auto_single_row_table_yields_nothing(self):
        # One row cannot share values with anything — no join hub possible.
        rows = _unique_rows([_row(row_data={"id": 1, "status": "active"})])
        assert _resolve_column_value_selection(rows, {"*": ["auto"]}) == {}


class TestSelectedColumnValues:
    ROW_DATA = {"id": 1, "status": "active", "country": "RS", "customer_id": 7}
    FOREIGN_KEYS = [{"column": "customer_id", "ref_table": "customers", "ref_column": "id"}]

    def test_no_columns_selected(self):
        row = _row(row_data=self.ROW_DATA)
        assert _selected_column_values(row, None) == {}
        assert _selected_column_values(row, set()) == {}

    def test_specific_columns(self):
        row = _row(row_data=self.ROW_DATA)
        assert _selected_column_values(row, {"status"}) == {"status": "active"}

    def test_all_columns_excludes_pk_and_fk(self):
        row = _row(row_data=self.ROW_DATA, foreign_keys=self.FOREIGN_KEYS)
        assert _selected_column_values(row, {ALL_COLUMNS}) == {
            "status": "active",
            "country": "RS",
        }

    def test_pk_and_fk_excluded_even_when_listed(self):
        row = _row(row_data=self.ROW_DATA, foreign_keys=self.FOREIGN_KEYS)
        assert _selected_column_values(row, {"id", "customer_id"}) == {}

    def test_null_empty_and_overlong_values_skipped(self):
        row = _row(
            row_data={
                "id": 1,
                "status": None,
                "note": "  ",
                "blob": "x" * (MAX_COLUMN_VALUE_LENGTH + 1),
                "country": "RS",
            }
        )
        assert _selected_column_values(row, {ALL_COLUMNS}) == {"country": "RS"}

    def test_values_coerced_to_stripped_strings(self):
        row = _row(row_data={"id": 1, "amount": 42, "status": " active "})
        assert _selected_column_values(row, {ALL_COLUMNS}) == {
            "amount": "42",
            "status": "active",
        }


@pytest.mark.asyncio
async def test_emit_creates_shared_column_value_nodes_and_edges(monkeypatch):
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(schema_graph_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": None}),
    )

    row_a, row_b = str(uuid4()), str(uuid4())
    row_records = [
        {
            "source_id": row_a,
            "table_name": "orders",
            "fk_references": [],
            "column_values": {"status": "active", "country": "RS"},
        },
        {
            "source_id": row_b,
            "table_name": "orders",
            "fk_references": [],
            "column_values": {"status": "active"},
        },
    ]

    await emit_dlt_schema_graph({}, row_records, ctx=None)

    added_nodes = graph.add_nodes.call_args.args[0]
    value_nodes = [node for node in added_nodes if isinstance(node, ColumnValue)]
    # "active" is shared by both rows -> one node; "RS" -> one node.
    assert len(value_nodes) == 2
    assert {node.name for node in value_nodes} == {
        "orders:status:active",
        "orders:country:RS",
    }

    added_edges = graph.add_edges.call_args.args[0]
    status_edges = [edge for edge in added_edges if edge[2] == "status"]
    assert len(status_edges) == 2
    shared_node_id = {edge[1] for edge in status_edges}
    assert len(shared_node_id) == 1, "both rows must point at the same value node"
    assert {edge[0] for edge in status_edges} == {row_a, row_b}


@pytest.mark.asyncio
async def test_emit_without_column_values_adds_no_value_nodes(monkeypatch):
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(schema_graph_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": None}),
    )

    row_records = [
        {"source_id": str(uuid4()), "table_name": "orders", "fk_references": []},
    ]

    await emit_dlt_schema_graph({}, row_records, ctx=None)

    # Only the is_row_of edge is produced; no nodes are added at all.
    graph.add_nodes.assert_not_called()
    added_edges = graph.add_edges.call_args.args[0]
    assert all(edge[2] == "is_row_of" for edge in added_edges)
