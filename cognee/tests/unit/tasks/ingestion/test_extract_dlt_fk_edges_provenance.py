import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.graph.methods as graph_methods
import cognee.tasks.ingestion.dlt_schema_graph as schema_graph_module
import cognee.tasks.ingestion.extract_dlt_fk_edges as dlt_module
from cognee.modules.pipelines.models import PipelineContext

# The graph package's __init__ re-exports the get_graph_engine function under
# the same name as its submodule, so the module must be resolved explicitly.
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)


def _ext_metadata():
    return {
        "source": "dlt",
        "table_name": "orders",
        "schema_info": [{"name": "id", "data_type": "bigint"}],
        "foreign_keys": [{"column": "customer_id", "ref_table": "customers", "ref_column": "id"}],
        "dlt_db_name": "shop",
        "fk_references": [
            {
                "target_data_id": str(uuid4()),
                "relationship_name": "references_customers",
                "target_table": "customers",
                "column": "customer_id",
            }
        ],
    }


def _wire_dlt_module(monkeypatch, graph, provenance_kwargs):
    """Patch extract_dlt_fk_edges' collaborators; return (upsert_nodes, upsert_edges)."""
    monkeypatch.setattr(dlt_module, "_is_dlt_data_point", lambda dp: True)
    monkeypatch.setattr(dlt_module, "parse_external_metadata", lambda doc: _ext_metadata())
    # Graph writes are delegated to the shared emitter in dlt_schema_graph.
    monkeypatch.setattr(schema_graph_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(
        schema_graph_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value=provenance_kwargs),
    )
    # get_graph_engine and upsert_nodes/upsert_edges are imported inside the
    # task function, so they must be patched at their source modules.
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    upsert_nodes = AsyncMock()
    upsert_edges = AsyncMock()
    monkeypatch.setattr(graph_methods, "upsert_nodes", upsert_nodes)
    monkeypatch.setattr(graph_methods, "upsert_edges", upsert_edges)
    return upsert_nodes, upsert_edges


def _ctx():
    return PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_extract_dlt_fk_edges_stamps_writes_and_skips_ledger_on_graph_provenance(
    monkeypatch,
):
    """Graph-provenance graph: schema nodes and FK edges are folded with the
    source ref and the relational ledger is skipped (mirrors add_data_points)."""
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    ctx = _ctx()
    upsert_nodes, upsert_edges = _wire_dlt_module(
        monkeypatch,
        graph,
        {"source_ref_key": "source-ref", "pipeline_run_id": "run-id"},
    )
    data_points = [SimpleNamespace(is_part_of=SimpleNamespace(id=uuid4()))]

    result = await dlt_module.extract_dlt_fk_edges(data_points, ctx=ctx)

    assert result == data_points
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
async def test_extract_dlt_fk_edges_writes_ledger_on_ledger_graph(monkeypatch):
    """Ledger graph: the fold is a no-op (no source ref), so the schema nodes
    and FK edges are recorded in the relational rollback ledger instead."""
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    ctx = _ctx()
    upsert_nodes, upsert_edges = _wire_dlt_module(
        monkeypatch,
        graph,
        {"source_ref_key": None, "pipeline_run_id": None},
    )
    data_points = [SimpleNamespace(is_part_of=SimpleNamespace(id=uuid4()))]

    await dlt_module.extract_dlt_fk_edges(data_points, ctx=ctx)

    assert graph.add_nodes.await_args.kwargs == {
        "source_ref_key": None,
        "pipeline_run_id": None,
    }
    upsert_nodes.assert_awaited_once()
    assert upsert_nodes.await_args.kwargs["pipeline_run_id"] == ctx.pipeline_run_id
    upsert_edges.assert_awaited_once()
    assert upsert_edges.await_args.kwargs["pipeline_run_id"] == ctx.pipeline_run_id
