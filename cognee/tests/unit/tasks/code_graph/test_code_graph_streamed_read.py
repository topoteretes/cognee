"""The code graph load looks up only the nodes and edges of its own repository.

The load used to hold ``get_graph_data()`` whole, chunk text included, for
its whole run. It now looks up each node batch's ids before writing them, the
existing edges of each edge batch's sources, and after the writes the
repository's code nodes and their edges. Nothing else in the graph is read,
and the batch size, which slices the node write and the lookups, changes no
outcome.
"""

import importlib
import json
import sys
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.tasks.code_graph.extract_code_graph import fact_node_id, load_code_graph
from cognee.tests.utils.graph_lookups import lookups_through_get_graph_data

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
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
    lookups_through_get_graph_data(engine)
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

    await load_code_graph(
        repo_path=REPO_PATH, snapshot_dir=snapshot_dir, node_set=["new-team"], **batch_sizes
    )

    stamped = engine.add_nodes.await_args.args[0][0].last_delta
    return {
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
async def test_batch_size_changes_no_outcome(tmp_path, monkeypatch):
    default = await _load(tmp_path, monkeypatch)
    one_at_a_time = await _load(tmp_path, monkeypatch, batch_size=1)

    for key in ("edges_added", "nodes_deleted", "edges_deleted", "tags_removed", "delta"):
        assert one_at_a_time[key] == default[key], key

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
async def test_lookups_never_touch_the_documents_next_to_the_code(tmp_path, monkeypatch):
    engine = _mock_engine(monkeypatch)
    requested_ids, requested_types = set(), set()

    def recording(lookup, collect):
        async def wrapper(*args):
            collect(*args)
            return await lookup(*args)

        return wrapper

    engine.get_nodes = recording(engine.get_nodes, requested_ids.update)
    engine.get_outgoing_edge_keys = recording(engine.get_outgoing_edge_keys, requested_ids.update)
    engine.get_node_ids_by_property = recording(
        engine.get_node_ids_by_property, lambda types, *_: requested_types.update(types)
    )
    monkeypatch.setattr(add_data_points_module, "add_data_points", AsyncMock())

    await load_code_graph(
        repo_path=REPO_PATH, snapshot_dir=_write_snapshot(tmp_path), node_set=["new-team"]
    )

    code_ids = {REPO_NODE_ID, DATABASE_ID, HANDLER_ID, _node_id("symbol", "app/api.new_route")}
    assert requested_ids <= code_ids | {STALE_ID, OLD_TAG_ID}
    assert {CHUNK_ID, DOCUMENT_ID}.isdisjoint(requested_ids)
    assert requested_types <= extract_module._code_graph_node_types()


@pytest.mark.asyncio
async def test_node_write_is_sliced_repository_first(tmp_path, monkeypatch):
    default = await _load(tmp_path, monkeypatch)
    sliced = await _load(tmp_path, monkeypatch, batch_size=1)

    assert len(default["written_slices"]) == 1
    assert [len(batch) for batch in sliced["written_slices"]] == [1, 1, 1, 1]
    assert sliced["written_slices"][0] == [REPO_NODE_ID]
    assert [node_id for batch in sliced["written_slices"] for node_id in batch] == default[
        "written_slices"
    ][0]
    assert sliced["delta"] == default["delta"]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1])
async def test_non_positive_batch_size_raises_before_reading(tmp_path, monkeypatch, batch_size):
    engine = _mock_engine(monkeypatch)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        await load_code_graph(
            repo_path=REPO_PATH, snapshot_dir=_write_snapshot(tmp_path), batch_size=batch_size
        )
    engine.get_graph_data.assert_not_awaited()
    engine.get_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_neptune_node_maps_give_the_same_outcome(tmp_path, monkeypatch):
    """Neptune's get_nodes returns openCypher node maps: ``~id`` and ``~properties``."""
    default = await _load(tmp_path, monkeypatch)
    flat_lookups = lookups_through_get_graph_data

    def neptune_lookups(engine):
        flat_lookups(engine)
        flat_get_nodes = engine.get_nodes

        async def get_nodes(node_ids):
            nodes = await flat_get_nodes(node_ids)
            return [{"~id": node.pop("id"), "~properties": node} for node in nodes]

        engine.get_nodes = get_nodes

    monkeypatch.setattr(sys.modules[__name__], "lookups_through_get_graph_data", neptune_lookups)
    neptune = await _load(tmp_path, monkeypatch)

    assert neptune == default


@pytest.mark.asyncio
async def test_neptune_node_map_gives_the_stored_repository_state(monkeypatch):
    """The skip check reads the repository node off Neptune's ``get_node`` map."""
    version = extract_module.CODE_GRAPH_MAPPING_VERSION
    engine = _mock_engine(monkeypatch)
    engine.get_node.return_value = {
        "~id": REPO_NODE_ID,
        "~labels": ["CodeRepository"],
        "~properties": {
            "name": REPO,
            "last_snapshot_id": "sha256:new",
            "last_mapping_version": version,
            "belongs_to_set": ["old-team"],
        },
    }

    state = await extract_module._stored_repository_state(REPO)

    assert state == ("sha256:new", version, True)


@pytest.mark.asyncio
async def test_an_adapter_without_the_lookups_fails_before_writing(tmp_path, monkeypatch):
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    class NoLookups(GraphDBInterface):
        pass

    NoLookups.__abstractmethods__ = frozenset()
    engine = NoLookups()
    engine.get_node = AsyncMock(return_value=None)
    # Through the handle get_graph_engine really returns: type() of it is the
    # handle, so the check must read the adapter's class through __class__.
    monkeypatch.setattr(graph_engine_module, "create_graph_engine", lambda **config: engine)
    handle = graph_engine_module._GraphEngineHandle({})
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=handle))
    write = AsyncMock()
    monkeypatch.setattr(add_data_points_module, "add_data_points", write)

    with pytest.raises(NotImplementedError, match="get_node_ids_by_property or get_outgoing"):
        await load_code_graph(repo_path=REPO_PATH, snapshot_dir=_write_snapshot(tmp_path))
    write.assert_not_awaited()
