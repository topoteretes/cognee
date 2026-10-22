"""emit_dlt_schema_graph: dual-provenance writes and UUID edge tuples.

Behaviors formerly pinned through the removed legacy per-row task — they
belong to the shared emitter, which the manifest pipeline
(extract_dlt_source_edges) drives:

- graph-provenance graphs get writes stamped with the source ref and the
  relational rollback ledger is skipped;
- ledger graphs record nodes/edges in the ledger with the run's id;
- edge tuples carry UUIDs in slots 0/1 — upsert_edges inserts them into
  UUID-typed columns, so stringified ids crash the ledger.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import cognee.modules.graph.methods as graph_methods
import cognee.tasks.ingestion.dlt_schema_graph as schema_graph_module
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.ingestion.dlt_schema_graph import emit_dlt_schema_graph

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)


def _tables():
    return {
        "orders": {
            "schema_info": [{"name": "id", "data_type": "bigint"}],
            "foreign_keys": [
                {"column": "customer_id", "ref_table": "customers", "ref_column": "id"}
            ],
            "primary_key": "id",
            "row_count": 1,
        }
    }


def _row_records():
    return [
        {
            "source_id": str(uuid4()),
            "table_name": "orders",
            "fk_references": [
                {
                    "target_node_id": str(uuid4()),
                    "relationship_name": "references_customers",
                    "target_table": "customers",
                    "column": "customer_id",
                }
            ],
            "column_values": {},
        }
    ]


def _ctx():
    return PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )


def _wire(monkeypatch, graph, provenance_kwargs):
    monkeypatch.setattr(schema_graph_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value=provenance_kwargs),
    )
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    upsert_nodes = AsyncMock()
    upsert_edges = AsyncMock()
    monkeypatch.setattr(graph_methods, "upsert_nodes", upsert_nodes)
    monkeypatch.setattr(graph_methods, "upsert_edges", upsert_edges)
    return upsert_nodes, upsert_edges


@pytest.mark.asyncio
async def test_emit_stamps_writes_and_skips_ledger_on_graph_provenance(monkeypatch):
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    upsert_nodes, upsert_edges = _wire(
        monkeypatch, graph, {"source_ref_key": "source-ref", "pipeline_run_id": "run-id"}
    )

    await emit_dlt_schema_graph(_tables(), _row_records(), _ctx())

    graph.add_nodes.assert_awaited_once()
    assert graph.add_nodes.await_args.kwargs == {
        "source_ref_key": "source-ref",
        "pipeline_run_id": "run-id",
    }
    graph.add_edges.assert_awaited_once()
    assert graph.add_edges.await_args.kwargs == {
        "source_ref_key": "source-ref",
        "pipeline_run_id": "run-id",
    }
    upsert_nodes.assert_not_awaited()
    upsert_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_emit_writes_ledger_on_ledger_graph(monkeypatch):
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    upsert_nodes, upsert_edges = _wire(
        monkeypatch, graph, {"source_ref_key": None, "pipeline_run_id": None}
    )
    ctx = _ctx()

    await emit_dlt_schema_graph(_tables(), _row_records(), ctx)

    assert graph.add_nodes.await_args.kwargs == {
        "source_ref_key": None,
        "pipeline_run_id": None,
    }
    upsert_nodes.assert_awaited_once()
    assert upsert_nodes.await_args.kwargs["pipeline_run_id"] == ctx.pipeline_run_id
    upsert_edges.assert_awaited_once()
    assert upsert_edges.await_args.kwargs["pipeline_run_id"] == ctx.pipeline_run_id


@pytest.mark.asyncio
async def test_emit_passes_uuid_node_ids_in_edge_tuples(monkeypatch):
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    _, upsert_edges = _wire(monkeypatch, graph, {"source_ref_key": None, "pipeline_run_id": None})

    await emit_dlt_schema_graph(_tables(), _row_records(), _ctx())

    graph_edges = graph.add_edges.await_args.args[0]
    ledger_edges = upsert_edges.await_args.args[0]
    assert graph_edges and ledger_edges
    for source_id, target_id, _name, _attrs in [*graph_edges, *ledger_edges]:
        assert isinstance(source_id, UUID), f"str source id would crash the ledger: {source_id!r}"
        assert isinstance(target_id, UUID), f"str target id would crash the ledger: {target_id!r}"


@pytest.mark.asyncio
async def test_tables_of_one_name_from_two_sources_are_separate_nodes(monkeypatch):
    """Each source's table node keeps its own keys, names its database, links to
    that database's source document, and its rows point at it."""
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    _wire(monkeypatch, graph, {"source_ref_key": None, "pipeline_run_id": None})
    shop, crm = str(uuid4()), str(uuid4())
    orders = _tables()["orders"]
    tables = {
        f"{shop}:orders": {
            **orders,
            "name": "orders",
            "scope": shop,
            "source_label": "shop.sqlite",
        },
        f"{crm}:orders": {
            **orders,
            "name": "orders",
            "scope": crm,
            "source_label": "crm",
            "foreign_keys": [],
        },
    }
    shop_row, crm_row = str(uuid4()), str(uuid4())
    row_records = [
        {"source_id": shop_row, "table_name": "orders", "scope": shop, "fk_references": []},
        {"source_id": crm_row, "table_name": "orders", "scope": crm, "fk_references": []},
    ]

    await emit_dlt_schema_graph(tables, row_records, _ctx())

    nodes = graph.add_nodes.await_args.args[0]
    table_nodes = {node.id: node for node in nodes if type(node).__name__ == "SchemaTable"}
    shop_table = schema_graph_module._table_node_id("orders", shop)
    crm_table = schema_graph_module._table_node_id("orders", crm)
    assert set(table_nodes) == {shop_table, crm_table}
    assert table_nodes[shop_table].foreign_keys != table_nodes[crm_table].foreign_keys
    assert table_nodes[shop_table].description == (
        "Relational table 'orders' from shop.sqlite with 1 rows, loaded through dlt."
    )
    assert "from crm with" in table_nodes[crm_table].description

    edges = graph.add_edges.await_args.args[0]
    row_of = {(str(source), target) for source, target, name, _ in edges if name == "is_row_of"}
    assert row_of == {(shop_row, shop_table), (crm_row, crm_table)}
    table_of = {(source, str(target)) for source, target, name, _ in edges if name == "is_table_of"}
    assert table_of == {(shop_table, shop), (crm_table, crm)}
