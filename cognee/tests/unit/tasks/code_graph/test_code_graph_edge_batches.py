"""The typed edges are resolved and written in batches, never held all at once.

``iter_code_graph_edges`` yields each edge as it resolves; add_code_graph_edges
writes the new ones ``write_batch_size`` at a time and records each batch in
the rollback ledger right after its write. The batch size changes neither
which edges are written nor what the sweep removes.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_edges,
    build_code_graph_edges,
    fact_node_id,
    iter_code_graph_edges,
)
from cognee.tests.utils.graph_stream import stream_through_get_graph_data

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
graph_methods_module = importlib.import_module("cognee.modules.graph.methods")

REPO = "demo_repo"
REPO_PATH = f"/repos/{REPO}"
NAMES = ["app.a", "app.b", "app.c", "app.d"]
FACTS = [
    {
        "kind": "symbol",
        "name": name,
        "file": "app.py",
        "repo": REPO,
        "relations": [{"kind": "calls", "target": target} for target in NAMES if target != name]
        + [{"kind": "calls", "target": "missing.symbol"}],
    }
    for name in NAMES
]
# Four symbols each calling the other three: twelve edges, four skipped.
EDGE_COUNT = 12


def _id(name):
    return str(fact_node_id(REPO, "symbol", name))


def _write_snapshot(tmp_path):
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in FACTS) + "\n")
    (tmp_path / "receipt.json").write_text(json.dumps({"snapshot_id": "sha256:edges"}))
    return tmp_path


def _mock_engine(monkeypatch, nodes=(), edges=()):
    engine = AsyncMock()
    engine.get_graph_data.return_value = (list(nodes), list(edges))
    stream_through_get_graph_data(engine)
    engine.get_node.return_value = None
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    return engine


def _written(engine):
    return [
        [(str(edge[0]), str(edge[1]), edge[2]) for edge in call.args[0]]
        for call in engine.add_edges.await_args_list
    ]


def _ctx():
    return SimpleNamespace(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        pipeline_run_id=uuid4(),
    )


def test_iterator_yields_what_the_list_holds_and_returns_the_skips():
    edges, skipped = build_code_graph_edges(FACTS, repo_path=REPO_PATH)

    resolver = iter_code_graph_edges(FACTS, repo_path=REPO_PATH)
    streamed = []
    while True:
        try:
            streamed.append(next(resolver))
        except StopIteration as stop:
            streamed_skipped = stop.value
            break

    assert [edge[:3] for edge in streamed] == [edge[:3] for edge in edges]
    assert len(edges) == EDGE_COUNT
    assert streamed_skipped == skipped == len(NAMES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write_batch_size", "expected_sizes"),
    [(1, [1] * 12), (5, [5, 5, 2]), (6, [6, 6]), (12, [12]), (2000, [12])],
)
async def test_new_edges_are_written_in_batches_of_the_write_batch_size(
    tmp_path, monkeypatch, write_batch_size, expected_sizes
):
    engine = _mock_engine(monkeypatch)
    snapshot_dir = _write_snapshot(tmp_path)
    unbatched, _skipped = build_code_graph_edges(FACTS, repo_path=REPO_PATH)

    await add_code_graph_edges(
        ["sentinel"],
        repo_path=REPO_PATH,
        snapshot_dir=snapshot_dir,
        write_batch_size=write_batch_size,
    )

    batches = _written(engine)
    assert [len(batch) for batch in batches] == expected_sizes
    assert [key for batch in batches for key in batch] == [
        (str(edge[0]), str(edge[1]), edge[2]) for edge in unbatched
    ]
    assert engine.add_nodes.await_args.args[0][0].last_delta["edges_added"] == EDGE_COUNT


@pytest.mark.asyncio
async def test_edges_already_in_the_graph_are_not_written_and_stale_ones_are_swept(
    tmp_path, monkeypatch
):
    """The diff and sweep see the whole pre-write state whatever the batch size."""
    present = (_id("app.a"), _id("app.b"), "calls")
    stale = (_id("app.a"), _id("app.b"), "imports")
    nodes = [(_id(name), {"type": "CodeSymbol", "name": name, "repo": REPO}) for name in NAMES]
    engine = _mock_engine(monkeypatch, nodes, [(*present, {}), (*stale, {})])

    await add_code_graph_edges(
        ["sentinel"],
        repo_path=REPO_PATH,
        snapshot_dir=_write_snapshot(tmp_path),
        write_batch_size=5,
    )

    written = [key for batch in _written(engine) for key in batch]
    assert present not in written
    assert [len(batch) for batch in _written(engine)] == [5, 5, 1]
    deleted = [
        (edge.source_id, edge.target_id, edge.relationship_name)
        for call in engine.delete_edge_triples.await_args_list
        for edge in call.args[0]
    ]
    assert deleted == [stale]
    assert engine.add_nodes.await_args.args[0][0].last_delta["edges_added"] == EDGE_COUNT - 1


@pytest.mark.asyncio
async def test_every_written_batch_is_recorded_in_the_ledger(tmp_path, monkeypatch):
    engine = _mock_engine(monkeypatch)
    ledger = AsyncMock()
    monkeypatch.setattr(graph_methods_module, "upsert_edges", ledger)
    ctx = _ctx()

    await add_code_graph_edges(
        ["sentinel"],
        repo_path=REPO_PATH,
        snapshot_dir=_write_snapshot(tmp_path),
        ctx=ctx,
        write_batch_size=5,
    )

    assert [call.args[0] for call in ledger.await_args_list] == [
        call.args[0] for call in engine.add_edges.await_args_list
    ]
    assert all(call.kwargs["data_id"] == ctx.data_item.id for call in ledger.await_args_list)
    assert all(
        call.kwargs["pipeline_run_id"] == ctx.pipeline_run_id for call in ledger.await_args_list
    )


@pytest.mark.asyncio
async def test_a_failed_batch_keeps_the_written_batches_in_the_ledger_and_raises(
    tmp_path, monkeypatch
):
    engine = _mock_engine(monkeypatch)
    engine.add_edges.side_effect = [None, RuntimeError("graph write failed")]
    ledger = AsyncMock()
    monkeypatch.setattr(graph_methods_module, "upsert_edges", ledger)

    with pytest.raises(RuntimeError, match="graph write failed"):
        await add_code_graph_edges(
            ["sentinel"],
            repo_path=REPO_PATH,
            snapshot_dir=_write_snapshot(tmp_path),
            ctx=_ctx(),
            write_batch_size=5,
        )

    assert ledger.await_count == 1
    assert len(ledger.await_args.args[0]) == 5
    # The run stops at the failure: no sweep and no snapshot stamp.
    engine.delete_edge_triples.assert_not_awaited()
    engine.add_nodes.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("write_batch_size", [0, -1])
async def test_non_positive_write_batch_size_raises_before_reading(
    tmp_path, monkeypatch, write_batch_size
):
    engine = _mock_engine(monkeypatch)

    with pytest.raises(ValueError, match="write_batch_size"):
        await add_code_graph_edges(
            ["sentinel"],
            repo_path=REPO_PATH,
            snapshot_dir=_write_snapshot(tmp_path),
            write_batch_size=write_batch_size,
        )
    engine.get_graph_data.assert_not_awaited()
