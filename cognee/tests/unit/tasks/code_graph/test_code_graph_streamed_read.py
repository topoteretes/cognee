"""The code graph load reads the graph as a stream and keeps only what it reads.

``add_code_graph_data_points`` used to hold ``get_graph_data()`` whole, chunk
text included, for the rest of the load. It now drains ``stream_graph_data``
into a trimmed state: every id and fact_hash, the code-graph and NodeSet nodes
with the four properties the sweep and detag read, and the edges that start
at a code-graph node. The page size must not change any outcome, and neither
must slicing the node write.
"""

import importlib
import json
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    fact_node_id,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import CodeRepository
from cognee.tests.utils.graph_stream import stream_through_get_graph_data

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

REPO = "demo_repo"
REPO_PATH = f"/repos/{REPO}"

FACTS = [
    {"kind": "symbol", "name": "app/db.Database", "file": "app/db.py", "repo": REPO},
    {
        "kind": "symbol",
        "name": "app/api.handler",
        "file": "app/api.py",
        "repo": REPO,
        "relations": [{"kind": "calls", "target": "app/db.Database"}],
    },
    {"kind": "symbol", "name": "app/api.new_route", "file": "app/api.py", "repo": REPO},
]


def _node_id(kind, name):
    return str(fact_node_id(REPO, kind, name))


REPO_NODE_ID = _node_id("repository", REPO)
DATABASE_ID = _node_id("symbol", "app/db.Database")
HANDLER_ID = _node_id("symbol", "app/api.handler")
STALE_ID = _node_id("symbol", "app/old.removed")
OLD_TAG_ID = str(generate_node_id("NodeSet:old-team"))
CHUNK_ID = "00000000-0000-0000-0000-00000000c4c4"
DOCUMENT_ID = "00000000-0000-0000-0000-00000000d0c0"
CHUNK_TEXT = "chunk text " * 10_000


def _existing_graph():
    """A graph from an earlier load, tagged old-team, next to a document's chunk."""
    nodes = [
        (
            REPO_NODE_ID,
            {
                "type": "CodeRepository",
                "name": REPO,
                "belongs_to_set": ["old-team"],
                "last_snapshot_id": "sha256:old",
                "description": "repository",
            },
        ),
        (
            DATABASE_ID,
            {
                "type": "CodeSymbol",
                "name": "app/db.Database",
                "repo": REPO,
                "fact_hash": "stale-hash",
                "belongs_to_set": ["old-team"],
                "description": "x" * 500,
            },
        ),
        (
            HANDLER_ID,
            {"type": "CodeSymbol", "name": "app/api.handler", "repo": REPO, "fact_hash": "h"},
        ),
        (STALE_ID, {"type": "CodeSymbol", "name": "app/old.removed", "repo": REPO}),
        (OLD_TAG_ID, {"type": "NodeSet", "name": "old-team"}),
        (CHUNK_ID, {"type": "DocumentChunk", "name": "chunk", "text": CHUNK_TEXT}),
        (DOCUMENT_ID, {"type": "TextDocument", "name": "notes.md"}),
    ]
    edges = [
        (DATABASE_ID, REPO_NODE_ID, "part_of", {}),
        (HANDLER_ID, REPO_NODE_ID, "part_of", {}),
        (STALE_ID, REPO_NODE_ID, "part_of", {}),
        (HANDLER_ID, DATABASE_ID, "calls", {}),
        # Stale: both endpoints survive, the snapshot no longer derives it.
        (DATABASE_ID, HANDLER_ID, "imports", {}),
        (DATABASE_ID, OLD_TAG_ID, "belongs_to_set", {}),
        (REPO_NODE_ID, OLD_TAG_ID, "belongs_to_set", {}),
        (CHUNK_ID, DOCUMENT_ID, "is_part_of", {}),
        (CHUNK_ID, HANDLER_ID, "describes", {}),
    ]
    return nodes, edges


def _write_snapshot(tmp_path):
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in FACTS) + "\n")
    (tmp_path / "receipt.json").write_text(json.dumps({"snapshot_id": "sha256:new"}))
    return tmp_path


def _mock_engine(monkeypatch):
    nodes, edges = _existing_graph()
    engine = AsyncMock()
    engine.get_graph_data.return_value = (nodes, edges)
    stream_through_get_graph_data(engine)
    engine.get_node.return_value = None
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    return engine


async def _load(tmp_path, monkeypatch, **batch_sizes):
    """One full load (nodes, then edges, sweep, detag and stamp); what it wrote and deleted."""
    engine = _mock_engine(monkeypatch)
    write = AsyncMock()
    monkeypatch.setattr(add_data_points_module, "add_data_points", write)
    snapshot_dir = _write_snapshot(tmp_path)
    data_points = map_facts_to_data_points(FACTS, repo_path=REPO_PATH, node_set=["new-team"])

    read_batch = {
        key: value for key, value in batch_sizes.items() if key == "graph_read_batch_size"
    }
    state = await add_code_graph_data_points(data_points, **batch_sizes)
    await add_code_graph_edges(
        state,
        repo_path=REPO_PATH,
        snapshot_dir=snapshot_dir,
        node_set=["new-team"],
        **read_batch,
    )

    stamped = engine.add_nodes.await_args.args[0][0].last_delta
    return {
        "state": state,
        "written_slices": [
            [str(point.id) for point in call.args[0]] for call in write.await_args_list
        ],
        "edges_added": sorted(
            (str(edge[0]), str(edge[1]), edge[2])
            for call in engine.add_edges.await_args_list
            for edge in call.args[0]
        ),
        "nodes_deleted": sorted(
            node_id for call in engine.delete_nodes.await_args_list for node_id in call.args[0]
        ),
        "edges_deleted": sorted(
            (edge.source_id, edge.target_id, edge.relationship_name)
            for call in engine.delete_edge_triples.await_args_list
            for edge in call.args[0]
        ),
        "tags_removed": [
            (call.args[0], sorted(call.kwargs["node_ids"]))
            for call in engine.remove_belongs_to_set_tags.await_args_list
        ],
        "delta": {key: value for key, value in stamped.items() if key != "loaded_at"},
    }


@pytest.mark.asyncio
async def test_read_page_size_changes_no_outcome(tmp_path, monkeypatch):
    default = await _load(tmp_path, monkeypatch)
    one_row_pages = await _load(tmp_path, monkeypatch, graph_read_batch_size=1)

    for key in ("edges_added", "nodes_deleted", "edges_deleted", "tags_removed", "delta"):
        assert one_row_pages[key] == default[key], key
    assert one_row_pages["state"].node_delta == default["state"].node_delta

    # The outcome itself, so equality is not two empty results.
    assert default["nodes_deleted"] == [STALE_ID]
    assert default["edges_deleted"] == sorted(
        [
            (DATABASE_ID, HANDLER_ID, "imports"),
            (DATABASE_ID, OLD_TAG_ID, "belongs_to_set"),
            (HANDLER_ID, OLD_TAG_ID, "belongs_to_set"),
            (REPO_NODE_ID, OLD_TAG_ID, "belongs_to_set"),
        ]
    )
    assert default["tags_removed"] == [
        (["old-team"], sorted([REPO_NODE_ID, DATABASE_ID, HANDLER_ID]))
    ]
    assert default["edges_added"] == []
    assert default["delta"]["nodes_added"] == 1
    assert default["delta"]["nodes_updated"] == 2


@pytest.mark.asyncio
async def test_state_keeps_no_chunk_content_or_chunk_edges(tmp_path, monkeypatch):
    state = (await _load(tmp_path, monkeypatch, graph_read_batch_size=2))["state"]

    kept_types = {properties.get("type") for _node_id, properties in state.existing_nodes}
    assert kept_types == {"CodeRepository", "CodeSymbol", "NodeSet"}
    assert all("text" not in properties for _node_id, properties in state.existing_nodes)
    assert all(
        set(properties) <= {"type", "name", "repo", "belongs_to_set"}
        for _node_id, properties in state.existing_nodes
    )
    assert {source for source, _target, _name in state.existing_edge_keys} == {
        REPO_NODE_ID,
        DATABASE_ID,
        HANDLER_ID,
        STALE_ID,
    }
    # Every node id is kept, for a cluster member's known ids.
    assert {CHUNK_ID, DOCUMENT_ID} <= state.existing_node_ids


@pytest.mark.asyncio
async def test_node_write_is_sliced_repository_first(tmp_path, monkeypatch):
    default = await _load(tmp_path, monkeypatch)
    sliced = await _load(tmp_path, monkeypatch, write_batch_size=1)

    assert len(default["written_slices"]) == 1
    assert [len(batch) for batch in sliced["written_slices"]] == [1, 1, 1, 1]
    assert sliced["written_slices"][0] == [REPO_NODE_ID]
    assert [node_id for batch in sliced["written_slices"] for node_id in batch] == default[
        "written_slices"
    ][0]
    assert sliced["delta"] == default["delta"]


@pytest.mark.asyncio
async def test_non_positive_write_batch_size_raises_before_reading(monkeypatch):
    engine = _mock_engine(monkeypatch)
    data_points = map_facts_to_data_points(FACTS, repo_path=REPO_PATH)
    assert isinstance(data_points[0], CodeRepository)

    with pytest.raises(ValueError, match="write_batch_size"):
        await add_code_graph_data_points(data_points, write_batch_size=0)
    engine.get_graph_data.assert_not_awaited()
