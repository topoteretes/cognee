"""Regression: DLT FK/schema edge tuples must carry UUID node ids.

``upsert_edges`` declares ``List[Tuple[UUID, UUID, str, Dict]]`` and inserts
slots 0/1 straight into UUID-typed relational columns, so stringified ids
crash the rollback ledger with "'str' object has no attribute 'hex'" — for
every DLT source, since the ``is_row_of`` edge fires per row even without FKs.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

import cognee.modules.graph.methods as graph_methods
import cognee.tasks.ingestion.extract_dlt_fk_edges as dlt_module
from cognee.modules.pipelines.models import PipelineContext

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)


@pytest.mark.asyncio
async def test_extract_dlt_fk_edges_passes_uuid_node_ids(monkeypatch):
    ext_metadata = {
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
    graph = SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock())
    monkeypatch.setattr(dlt_module, "_is_dlt_data_point", lambda dp: True)
    monkeypatch.setattr(dlt_module, "parse_external_metadata", lambda doc: ext_metadata)
    monkeypatch.setattr(dlt_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(
        dlt_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value={"source_ref_key": None, "pipeline_run_id": None}),
    )
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=graph))
    upsert_edges = AsyncMock()
    monkeypatch.setattr(graph_methods, "upsert_nodes", AsyncMock())
    monkeypatch.setattr(graph_methods, "upsert_edges", upsert_edges)
    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )

    await dlt_module.extract_dlt_fk_edges(
        [SimpleNamespace(is_part_of=SimpleNamespace(id=uuid4()))], ctx=ctx
    )

    graph_edges = graph.add_edges.await_args.args[0]
    ledger_edges = upsert_edges.await_args.args[0]
    assert graph_edges and ledger_edges
    for source_id, target_id, _name, _attrs in [*graph_edges, *ledger_edges]:
        assert isinstance(source_id, UUID), f"str source id would crash the ledger: {source_id!r}"
        assert isinstance(target_id, UUID), f"str target id would crash the ledger: {target_id!r}"
