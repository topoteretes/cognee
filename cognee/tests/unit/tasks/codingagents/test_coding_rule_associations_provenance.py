from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.tasks.codingagents.coding_rule_associations as rule_module
from cognee.modules.pipelines.models import PipelineContext


def _wire_rule_module(monkeypatch, graph, edge, provenance_kwargs):
    """Patch add_rule_associations' collaborators; return (add_data_points, upsert_edges)."""
    monkeypatch.setattr(rule_module, "get_graph_engine", AsyncMock(return_value=graph))
    monkeypatch.setattr(rule_module, "get_existing_rules", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        rule_module.LLMGateway,
        "acreate_structured_output",
        AsyncMock(
            return_value=rule_module.RuleSet(rules=[rule_module.Rule(text="Prefer clarity")])
        ),
    )
    monkeypatch.setattr(rule_module, "get_origin_edges", AsyncMock(return_value=[edge]))
    add_data_points = AsyncMock()
    monkeypatch.setattr(rule_module, "add_data_points", add_data_points)
    monkeypatch.setattr(
        rule_module,
        "graph_provenance_write_kwargs",
        AsyncMock(return_value=provenance_kwargs),
    )
    upsert_edges = AsyncMock()
    monkeypatch.setattr(rule_module, "upsert_edges", upsert_edges)
    monkeypatch.setattr(rule_module, "index_graph_edges", AsyncMock())
    return add_data_points, upsert_edges


def _ctx():
    return PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_add_rule_associations_stamps_edges_and_skips_ledger_on_graph_provenance(monkeypatch):
    """Graph-provenance graph: edges are folded with the source ref and the
    relational ledger is skipped (mirrors add_data_points)."""
    graph = SimpleNamespace(add_edges=AsyncMock())
    edge = ("rule-id", "chunk-id", "rule_associated_from", {})
    ctx = _ctx()
    add_data_points, upsert_edges = _wire_rule_module(
        monkeypatch,
        graph,
        edge,
        {"source_ref_key": "source-ref", "pipeline_run_id": "run-id"},
    )

    await rule_module.add_rule_associations("please be clear", "rules", ctx=ctx)

    add_data_points.assert_awaited_once()
    assert add_data_points.await_args.kwargs["ctx"] is ctx
    graph.add_edges.assert_awaited_once_with(
        [edge], source_ref_key="source-ref", pipeline_run_id="run-id"
    )
    upsert_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_rule_associations_writes_ledger_on_ledger_graph(monkeypatch):
    """Ledger graph: the fold is a no-op (no source ref), so the edges are
    recorded in the relational rollback ledger instead."""
    graph = SimpleNamespace(add_edges=AsyncMock())
    edge = ("rule-id", "chunk-id", "rule_associated_from", {})
    ctx = _ctx()
    _add_data_points, upsert_edges = _wire_rule_module(
        monkeypatch,
        graph,
        edge,
        {"source_ref_key": None, "pipeline_run_id": None},
    )

    await rule_module.add_rule_associations("please be clear", "rules", ctx=ctx)

    graph.add_edges.assert_awaited_once_with([edge], source_ref_key=None, pipeline_run_id=None)
    upsert_edges.assert_awaited_once()
    assert upsert_edges.await_args.kwargs["pipeline_run_id"] == ctx.pipeline_run_id
