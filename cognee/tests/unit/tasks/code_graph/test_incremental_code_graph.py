"""Incremental code-graph ingestion: snapshot-id skip and stale sweep (COG-6115)."""

import importlib
import json
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.code_graph.enola import snapshot_identity
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    extract_code_graph,
    fact_node_id,
)

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

REPO = "demo_repo"
SNAPSHOT_ID = "sha256:feedface"


def _write_snapshot(tmp_path, facts, snapshot_id=SNAPSHOT_ID):
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in facts) + "\n")
    (tmp_path / "receipt.json").write_text(json.dumps({"snapshot_id": snapshot_id}))
    return tmp_path


def _mock_engine(monkeypatch, nodes=(), edges=()):
    engine = AsyncMock()
    engine.get_graph_data.return_value = (list(nodes), list(edges))
    engine.get_node.return_value = None
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


FACTS = [
    {"kind": "symbol", "name": "app/db.Database", "file": "app/db.py", "repo": REPO},
    {
        "kind": "symbol",
        "name": "app/api.handler",
        "file": "app/api.py",
        "repo": REPO,
        "relations": [{"kind": "calls", "target": "app/db.Database"}],
    },
]


def _node_id(kind, name):
    return str(fact_node_id(REPO, kind, name))


REPO_NODE_ID = _node_id("repository", REPO)


def test_snapshot_identity_prefers_receipt_and_falls_back_to_hash(tmp_path):
    _write_snapshot(tmp_path, FACTS)
    assert snapshot_identity(tmp_path, {"snapshot_id": "sha256:abc"}) == "sha256:abc"

    hashed = snapshot_identity(tmp_path, None)
    assert hashed is not None and hashed.startswith("sha256:")
    assert snapshot_identity(tmp_path, {"snapshot_id": ""}) == hashed

    assert snapshot_identity(tmp_path / "missing", None) is None


@pytest.mark.asyncio
async def test_extract_skips_when_stored_snapshot_id_matches(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    engine = _mock_engine(monkeypatch)
    engine.get_node.return_value = {"last_snapshot_id": SNAPSHOT_ID}

    data_points = await extract_code_graph(repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)

    assert data_points == []
    engine.get_node.assert_awaited_once_with(REPO_NODE_ID)


@pytest.mark.asyncio
async def test_extract_reads_marker_from_json_properties_blob(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    engine = _mock_engine(monkeypatch)
    engine.get_node.return_value = {"properties": json.dumps({"last_snapshot_id": SNAPSHOT_ID})}

    data_points = await extract_code_graph(repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)

    assert data_points == []


@pytest.mark.asyncio
async def test_extract_loads_when_snapshot_id_differs_or_lookup_fails(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    engine = _mock_engine(monkeypatch)
    engine.get_node.return_value = {"last_snapshot_id": "sha256:older"}

    data_points = await extract_code_graph(repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)
    assert len(data_points) == 3  # repository + two symbols

    # A broken lookup must never block ingestion.
    engine.get_node.side_effect = RuntimeError("graph down")
    data_points = await extract_code_graph(repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)
    assert len(data_points) == 3


@pytest.mark.asyncio
async def test_skipped_snapshot_short_circuits_downstream_tasks(tmp_path, monkeypatch):
    engine = _mock_engine(monkeypatch)

    assert await add_code_graph_data_points([]) == []
    assert await add_code_graph_edges([], snapshot_dir=tmp_path) == []
    engine.add_edges.assert_not_awaited()
    engine.get_graph_data.assert_not_awaited()
    engine.add_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_removes_stale_nodes_and_edges_and_stamps_snapshot(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    stale_symbol_id = _node_id("symbol", "app/legacy.OldHelper")
    handler_id = _node_id("symbol", "app/api.handler")
    database_id = _node_id("symbol", "app/db.Database")
    nodeset_id = "11111111-1111-1111-1111-111111111111"

    existing_nodes = [
        (REPO_NODE_ID, {"type": "CodeRepository", "name": REPO}),
        (database_id, {"type": "CodeSymbol", "repo": REPO}),
        (handler_id, {"type": "CodeSymbol", "repo": REPO}),
        # Removed from the codebase -> must be swept.
        (stale_symbol_id, {"type": "CodeSymbol", "repo": REPO}),
        # Same type but another repo -> out of this snapshot's scope.
        (_node_id("symbol", "elsewhere"), {"type": "CodeSymbol", "repo": "other_repo"}),
        # Non-code node -> never swept.
        (nodeset_id, {"type": "NodeSet", "name": "code"}),
    ]
    existing_edges = [
        (database_id, REPO_NODE_ID, "part_of", {}),
        (handler_id, REPO_NODE_ID, "part_of", {}),
        (handler_id, database_id, "calls", {}),
        # The dependency was removed from the code -> stale, must be swept.
        (database_id, handler_id, "calls", {}),
        # Edge to a non-code node -> must survive.
        (handler_id, nodeset_id, "belongs_to_set", {}),
    ]
    engine = _mock_engine(monkeypatch, existing_nodes, existing_edges)
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )

    await add_code_graph_edges(["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)

    deleted_nodes = [
        node_id for call in engine.delete_nodes.await_args_list for node_id in call.args[0]
    ]
    assert deleted_nodes == [stale_symbol_id]

    deleted_edges = [
        edge for call in engine.delete_edge_triples.await_args_list for edge in call.args[0]
    ]
    assert [(e.source_id, e.target_id, e.relationship_name) for e in deleted_edges] == [
        (database_id, handler_id, "calls")
    ]

    stamped = engine.add_nodes.await_args.args[0]
    assert [node.name for node in stamped] == [REPO]
    assert stamped[0].last_snapshot_id == SNAPSHOT_ID


@pytest.mark.asyncio
async def test_sweep_failure_prevents_snapshot_stamp(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, FACTS)
    engine = _mock_engine(monkeypatch)
    engine.get_graph_data.side_effect = RuntimeError("graph down")
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )

    with pytest.raises(RuntimeError):
        await add_code_graph_edges(["sentinel"], repo_path=f"/repos/{REPO}", snapshot_dir=tmp_path)

    engine.add_nodes.assert_not_awaited()
