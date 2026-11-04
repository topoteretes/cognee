"""The snapshot is streamed: batches of DataPoints, then one end marker.

``extract_code_graph`` reads facts.jsonl line by line and yields lists of at
most ``batch_size`` DataPoints, then a ``_SnapshotEnd``. ``load_code_graph``
reads the graph once, before the first write, stores every batch against that
one read, and runs the edge step (edges, sweep, detag, stamp) once, on the
marker. An unchanged snapshot streams nothing, so nothing is read or written.
"""

import importlib
import json
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.code_graph.enola import EnolaSnapshot, parse_enola_snapshot
from cognee.tasks.code_graph.extract_code_graph import (
    CODE_GRAPH_MAPPING_VERSION,
    _SnapshotEnd,
    extract_code_graph,
    fact_node_id,
    load_code_graph,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import CodeRepository
from cognee.tests.utils.graph_stream import stream_through_get_graph_data

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
storage_module = importlib.import_module("cognee.tasks.storage.add_data_points")
extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")

REPO = "demo_repo"
REPO_PATH = f"/repos/{REPO}"
SNAPSHOT_ID = "sha256:stream"
NAMES = [f"app.f{index}" for index in range(5)]
FACTS = [
    {
        "kind": "symbol",
        "name": name,
        "file": "app.py",
        "repo": REPO,
        "relations": [{"kind": "calls", "target": NAMES[(index + 1) % len(NAMES)]}],
    }
    for index, name in enumerate(NAMES)
]


def _write_snapshot(tmp_path, facts=FACTS):
    lines = [json.dumps(fact) for fact in facts]
    # A blank and a corrupt line are skipped on every read, as parse_enola_snapshot skips them.
    (tmp_path / "facts.jsonl").write_text("\n".join(lines[:2] + ["", "{oops"] + lines[2:]) + "\n")
    (tmp_path / "receipt.json").write_text(json.dumps({"snapshot_id": SNAPSHOT_ID}))
    return tmp_path


def _mock_engine(monkeypatch, stored_repo=None):
    engine = AsyncMock()
    engine.get_graph_data.return_value = ([], [])
    stream_through_get_graph_data(engine)
    engine.get_node.return_value = stored_repo
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    return engine


async def _stream(**kwargs):
    return [item async for item in extract_code_graph(**kwargs)]


def test_streamed_snapshot_reads_the_facts_parse_enola_snapshot_reads(tmp_path):
    _write_snapshot(tmp_path)

    snapshot = EnolaSnapshot(tmp_path)

    facts, receipt = parse_enola_snapshot(tmp_path)
    assert list(snapshot.iter_facts()) == facts == FACTS
    # Read again on every call, never held.
    assert list(snapshot.iter_facts()) == facts
    assert snapshot.fact_count == len(FACTS)
    assert snapshot.receipt == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("batch_size", "sizes"), [(1, [1, 1, 1, 1, 1, 1]), (2, [2, 2, 2]), (4, [4, 2]), (6, [6])]
)
async def test_batches_of_batch_size_then_one_end_marker(tmp_path, monkeypatch, batch_size, sizes):
    _mock_engine(monkeypatch)

    items = await _stream(
        repo_path=REPO_PATH, snapshot_dir=_write_snapshot(tmp_path), batch_size=batch_size
    )

    *batches, end = items
    assert isinstance(end, _SnapshotEnd)
    assert not any(isinstance(batch, _SnapshotEnd) for batch in batches)
    assert [len(batch) for batch in batches] == sizes
    streamed = [point for batch in batches for point in batch]
    # The primary repository first, then the facts: map_facts_to_data_points' order.
    assert isinstance(streamed[0], CodeRepository)
    assert [point.id for point in streamed] == [
        point.id for point in map_facts_to_data_points(FACTS, repo_path=REPO_PATH)
    ]
    assert (end.repo_path, end.repo_scope, end.fallback_repo, end.snapshot_id) == (
        REPO_PATH,
        None,
        REPO,
        SNAPSHOT_ID,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1])
async def test_a_non_positive_batch_size_raises_before_anything_runs(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        await _stream(snapshot_dir=tmp_path, batch_size=batch_size)


@pytest.mark.asyncio
async def test_an_unchanged_snapshot_streams_nothing_and_the_load_writes_nothing(
    tmp_path, monkeypatch
):
    stored = {"last_snapshot_id": SNAPSHOT_ID, "last_mapping_version": CODE_GRAPH_MAPPING_VERSION}
    engine = _mock_engine(monkeypatch, stored_repo=stored)
    snapshot_dir = _write_snapshot(tmp_path)

    assert await _stream(repo_path=REPO_PATH, snapshot_dir=snapshot_dir) == []

    write = AsyncMock()
    monkeypatch.setattr(storage_module, "add_data_points", write)
    await load_code_graph(repo_path=REPO_PATH, snapshot_dir=snapshot_dir)

    write.assert_not_awaited()
    engine.get_graph_data.assert_not_awaited()
    engine.add_edges.assert_not_awaited()
    engine.add_nodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_graph_is_read_once_and_the_edge_step_runs_once_after_the_last_batch(
    tmp_path, monkeypatch
):
    engine = _mock_engine(monkeypatch)
    events = []

    async def write(points, **kwargs):
        events.append(("write", len(points)))

    async def add_edges(edges):
        events.append(("edges", len(edges)))

    async def add_nodes(nodes):
        events.append(("stamp", [node.name for node in nodes]))

    async def read(stream_engine, batch_size):
        events.append(("read", batch_size))
        return await original_read(stream_engine, batch_size)

    original_read = extract_module._read_code_graph_state
    monkeypatch.setattr(extract_module, "_read_code_graph_state", read)
    monkeypatch.setattr(storage_module, "add_data_points", write)
    engine.add_edges.side_effect = add_edges
    engine.add_nodes.side_effect = add_nodes

    await load_code_graph(
        repo_path=REPO_PATH,
        snapshot_dir=_write_snapshot(tmp_path),
        batch_size=2,
        graph_read_batch_size=7,
    )

    assert events == [
        ("read", 7),
        ("write", 2),
        ("write", 2),
        ("write", 2),
        ("edges", 2),
        ("edges", 2),
        ("edges", 1),
        ("stamp", [REPO]),
    ]
    stamped = engine.add_nodes.await_args.args[0][0]
    assert stamped.id == fact_node_id(REPO, "repository", REPO)
    assert stamped.last_delta["nodes_added"] == len(FACTS)
    assert stamped.last_delta["edges_added"] == len(FACTS)


@pytest.mark.asyncio
async def test_a_cluster_snapshot_held_in_memory_writes_its_resolved_edges_in_batches(
    tmp_path, monkeypatch
):
    """The one non-streamed case: a cluster's snapshot is parsed and its edges
    resolved once for every member; a member's load only batches them."""
    from cognee.tasks.code_graph.enola import ParsedSnapshot
    from cognee.tasks.code_graph.extract_code_graph import build_code_graph_edges

    engine = _mock_engine(monkeypatch)
    monkeypatch.setattr(storage_module, "add_data_points", AsyncMock())

    def _must_not_resolve(*args, **kwargs):
        raise AssertionError("a cluster member resolved the shared edges again")

    edges, _skipped = build_code_graph_edges(FACTS, repo_path=REPO_PATH)
    monkeypatch.setattr(extract_module, "iter_code_graph_edges", _must_not_resolve)
    snapshot = ParsedSnapshot(
        snapshot_dir=_write_snapshot(tmp_path), facts=FACTS, receipt=None, edges=edges
    )

    await load_code_graph(repo_path=REPO_PATH, snapshot=snapshot, batch_size=3)

    written = [call.args[0] for call in engine.add_edges.await_args_list]
    assert [len(batch) for batch in written] == [3, 2]
    assert [edge[:3] for batch in written for edge in batch] == [edge[:3] for edge in edges]
