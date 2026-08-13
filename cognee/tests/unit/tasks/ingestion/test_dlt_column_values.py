"""Tests for optional ColumnValue node emission in the DLT pipeline.

Column selection (resolve_dlt_sources._selected_column_values) supports
per-table column lists with "*" wildcards on either side; emission
(emit_dlt_schema_graph) creates one shared ColumnValue node per unique
(table, column, value) with a column-named edge from each row.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import cognee.tasks.ingestion.dlt_schema_graph as schema_graph_module
from cognee.modules.engine.models import ColumnValue
from cognee.tasks.ingestion.dlt_row_data import DltRowData
from cognee.tasks.ingestion.dlt_schema_graph import emit_dlt_schema_graph
from cognee.tasks.ingestion.resolve_dlt_sources import (
    MAX_COLUMN_VALUE_LENGTH,
    _selected_column_values,
)

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)


def _row(table_name="orders", row_data=None, foreign_keys=None, primary_key_column="id"):
    return DltRowData(
        table_name=table_name,
        primary_key_column=primary_key_column,
        primary_key_value="1",
        row_data=row_data or {},
        content_hash="hash",
        schema_info=[],
        schema_hash="schema-hash",
        foreign_keys=foreign_keys or [],
        dlt_db_name="shop",
        dataset_name="ds",
    )


class TestSelectedColumnValues:
    ROW_DATA = {"id": 1, "status": "active", "country": "RS", "customer_id": 7}
    FOREIGN_KEYS = [{"column": "customer_id", "ref_table": "customers", "ref_column": "id"}]

    def test_disabled_by_default(self):
        row = _row(row_data=self.ROW_DATA)
        assert _selected_column_values(row, {}) == {}
        assert _selected_column_values(row, None) == {}

    def test_specific_columns(self):
        row = _row(row_data=self.ROW_DATA)
        assert _selected_column_values(row, {"orders": ["status"]}) == {"status": "active"}

    def test_table_not_selected(self):
        row = _row(row_data=self.ROW_DATA)
        assert _selected_column_values(row, {"customers": ["status"]}) == {}

    def test_wildcard_table(self):
        row = _row(row_data=self.ROW_DATA)
        assert _selected_column_values(row, {"*": ["country"]}) == {"country": "RS"}

    def test_wildcard_columns_excludes_pk_and_fk(self):
        row = _row(row_data=self.ROW_DATA, foreign_keys=self.FOREIGN_KEYS)
        assert _selected_column_values(row, {"orders": ["*"]}) == {
            "status": "active",
            "country": "RS",
        }

    def test_all_tables_all_columns(self):
        row = _row(row_data=self.ROW_DATA, foreign_keys=self.FOREIGN_KEYS)
        assert _selected_column_values(row, {"*": ["*"]}) == {
            "status": "active",
            "country": "RS",
        }

    def test_pk_and_fk_excluded_even_when_listed(self):
        row = _row(row_data=self.ROW_DATA, foreign_keys=self.FOREIGN_KEYS)
        assert _selected_column_values(row, {"orders": ["id", "customer_id"]}) == {}

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
        assert _selected_column_values(row, {"orders": ["*"]}) == {"country": "RS"}

    def test_values_coerced_to_stripped_strings(self):
        row = _row(row_data={"id": 1, "amount": 42, "status": " active "})
        assert _selected_column_values(row, {"orders": ["*"]}) == {
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
    # Edge tuple slots carry UUIDs (the upsert_edges ledger contract).
    assert {edge[0] for edge in status_edges} == {UUID(row_a), UUID(row_b)}


@pytest.mark.asyncio
async def test_shared_set_prevents_reembedding_across_batches(monkeypatch):
    """A value seen by an earlier batch is not re-persisted or re-embedded,
    but every later row still gets its edge to the shared node."""
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    index_mock = AsyncMock()
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(schema_graph_module, "index_data_points", index_mock)
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": None}),
    )

    emitted_value_node_ids = set()
    row_a, row_b = str(uuid4()), str(uuid4())
    for source_id in (row_a, row_b):  # two batches, same shared value
        await emit_dlt_schema_graph(
            {},
            [
                {
                    "source_id": source_id,
                    "table_name": "orders",
                    "fk_references": [],
                    "column_values": {"status": "active"},
                }
            ],
            ctx=None,
            emitted_value_node_ids=emitted_value_node_ids,
        )

    # The value node is persisted and embedded exactly once (first batch).
    indexed = [n for call in index_mock.call_args_list for n in call.args[0]]
    assert len([n for n in indexed if isinstance(n, ColumnValue)]) == 1
    added = [n for call in graph.add_nodes.call_args_list for n in call.args[0]]
    assert len([n for n in added if isinstance(n, ColumnValue)]) == 1
    assert len(emitted_value_node_ids) == 1

    # Both batches still emitted their row edge to the same shared node.
    edges = [e for call in graph.add_edges.call_args_list for e in call.args[0]]
    status_edges = [e for e in edges if e[2] == "status"]
    assert {e[0] for e in status_edges} == {UUID(row_a), UUID(row_b)}
    assert len({e[1] for e in status_edges}) == 1


@pytest.mark.asyncio
async def test_emit_without_shared_set_keeps_per_call_behavior(monkeypatch):
    """Without the shared set (direct task invocation), each call persists."""
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    index_mock = AsyncMock()
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(schema_graph_module, "index_data_points", index_mock)
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": None}),
    )

    for _ in range(2):
        await emit_dlt_schema_graph(
            {},
            [
                {
                    "source_id": str(uuid4()),
                    "table_name": "orders",
                    "fk_references": [],
                    "column_values": {"status": "active"},
                }
            ],
            ctx=None,
        )

    indexed = [n for call in index_mock.call_args_list for n in call.args[0]]
    # Idempotent upsert by deterministic id — persisted per call when untracked.
    assert len([n for n in indexed if isinstance(n, ColumnValue)]) == 2


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
