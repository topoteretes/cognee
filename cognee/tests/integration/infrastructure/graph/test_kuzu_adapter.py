"""Tests for KuzuAdapter in both local and subprocess-proxy modes.

The subprocess mode uses ``get_graph_engine`` with
``graph_database_subprocess_enabled=True`` so the native ``kuzu.Database`` /
``kuzu.Connection`` live in a worker process while the adapter stays in the
main process.
"""

import json
import os
from pathlib import Path

import pytest
import pytest_asyncio

from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine

DEMO_KG_PATH = os.path.join(os.path.dirname(__file__), "test_kg.json")


def _load_demo_kg() -> KnowledgeGraph:
    data = json.loads(Path(DEMO_KG_PATH).read_text(encoding="utf-8"))
    return KnowledgeGraph.model_validate(data)


async def _close_adapter(a) -> None:
    """Tear-down helper. ``close`` is async and ``subprocess_enabled``
    adapters spawn a child process that must be reaped, or the pytest run
    leaks one worker per parametrized test.
    """
    try:
        await a.close()
    except Exception:
        pass


@pytest_asyncio.fixture
async def kuzu_adapter(tmp_path):
    """Direct KuzuAdapter instance (local mode)."""
    a = KuzuAdapter(db_path=str(tmp_path / "kuzu_direct"))
    if hasattr(a, "initialize"):
        await a.initialize()
    try:
        yield a
    finally:
        await _close_adapter(a)


@pytest_asyncio.fixture
async def subprocess_adapter(tmp_path):
    """KuzuAdapter backed by a subprocess-resident Kuzu client."""
    a = create_graph_engine(
        graph_database_provider="kuzu",
        graph_file_path=str(tmp_path / "kuzu_subprocess"),
        graph_database_subprocess_enabled=True,
    )
    if hasattr(a, "initialize"):
        await a.initialize()
    try:
        yield a
    finally:
        await _close_adapter(a)


@pytest_asyncio.fixture(params=["direct", "subprocess"])
async def adapter(request, tmp_path):
    """Parametrized fixture: both local and subprocess-backed adapters."""
    if request.param == "direct":
        a = KuzuAdapter(db_path=str(tmp_path / "kuzu_direct"))
    else:
        a = create_graph_engine(
            graph_database_provider="kuzu",
            graph_file_path=str(tmp_path / "kuzu_subprocess"),
            graph_database_subprocess_enabled=True,
        )
    if hasattr(a, "initialize"):
        await a.initialize()
    try:
        yield a
    finally:
        await _close_adapter(a)


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_empty_on_fresh_db(adapter):
    assert await adapter.is_empty() is True


# ---------------------------------------------------------------------------
# add_nodes (single) / has_node / get_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_get_node(adapter):
    kg = _load_demo_kg()
    node = kg.nodes[0]  # Alice

    # Use add_nodes (batch) with a single node — add_node uses a
    # different Cypher path that has a known Kuzu compatibility issue.
    await adapter.add_nodes([node])

    result = await adapter.get_node(node.id)
    assert result is not None
    assert result["id"] == node.id
    assert result["name"] == node.name


@pytest.mark.asyncio
async def test_has_node(adapter):
    kg = _load_demo_kg()
    node = kg.nodes[0]

    assert await adapter.has_node(node.id) is False
    await adapter.add_nodes([node])
    assert await adapter.has_node(node.id) is True


# ---------------------------------------------------------------------------
# add_nodes / get_nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_get_nodes(adapter):
    kg = _load_demo_kg()

    await adapter.add_nodes(kg.nodes)

    node_ids = [n.id for n in kg.nodes]
    results = await adapter.get_nodes(node_ids)
    assert len(results) == len(kg.nodes)

    result_ids = {r["id"] for r in results}
    assert result_ids == set(node_ids)


# ---------------------------------------------------------------------------
# delete_node / delete_nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_node(adapter):
    kg = _load_demo_kg()
    node = kg.nodes[0]

    await adapter.add_nodes([node])
    assert await adapter.has_node(node.id) is True

    await adapter.delete_node(node.id)
    assert await adapter.has_node(node.id) is False


@pytest.mark.asyncio
async def test_delete_nodes(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    ids = [n.id for n in kg.nodes]
    await adapter.delete_nodes(ids)

    assert await adapter.is_empty() is True


# ---------------------------------------------------------------------------
# add_edge / has_edge / get_edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_has_edge(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge = kg.edges[0]  # Alice -> Mark, "knows"
    await adapter.add_edge(edge.source_node_id, edge.target_node_id, edge.relationship_name, {})

    assert await adapter.has_edge(edge.source_node_id, edge.target_node_id, edge.relationship_name)


@pytest.mark.asyncio
async def test_get_edges(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge = kg.edges[0]
    await adapter.add_edge(edge.source_node_id, edge.target_node_id, edge.relationship_name, {})

    edges = await adapter.get_edges(edge.source_node_id)
    assert len(edges) >= 1


# ---------------------------------------------------------------------------
# add_edges / has_edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_edges_batch(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    # Verify all edges exist
    check_edges = [(e.source_node_id, e.target_node_id, e.relationship_name) for e in kg.edges]
    existing = await adapter.has_edges(check_edges)
    assert len(existing) == len(kg.edges)


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_neighbors(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    # Mark has edges: Alice->Mark (knows), Mark->Bob (had_dinner_with), Mark->Alice (had_dinner_with)
    neighbors = await adapter.get_neighbors("Mark")
    neighbor_ids = {n["id"] for n in neighbors}
    assert "Alice" in neighbor_ids
    assert "Bob" in neighbor_ids


# ---------------------------------------------------------------------------
# get_connections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_connections(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    connections = await adapter.get_connections("Mark")
    assert len(connections) >= 1

    # Each connection is (source_node, relationship_info, target_node)
    for conn in connections:
        assert len(conn) == 3


# ---------------------------------------------------------------------------
# get_graph_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_graph_data(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == len(kg.nodes)
    assert len(edges) == len(kg.edges)


# ---------------------------------------------------------------------------
# get_filtered_graph_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_filtered_graph_data(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    # Filter by type = "Person"
    nodes, edges = await adapter.get_filtered_graph_data([{"type": ["Person"]}])
    assert len(nodes) == len(kg.nodes)  # All nodes are Person type


# ---------------------------------------------------------------------------
# get_predecessors / get_successors
# Known adapter bug: RETURN properties(m) fails with current Kuzu version.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="KuzuAdapter bug: RETURN properties(m) not supported in current Kuzu")
async def test_predecessors_and_successors(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    # Alice->Mark (knows), so Mark's predecessors with "knows" should include Alice
    predecessors = await adapter.get_predecessors("Mark", edge_label="knows")
    assert len(predecessors) > 0

    # Mark->Bob (had_dinner_with), so Mark's successors should include Bob
    successors = await adapter.get_successors("Mark", edge_label="had_dinner_with")
    assert len(successors) > 0


# ---------------------------------------------------------------------------
# get_graph_metrics
# Known adapter bug: get_model_independent_graph_data returns wrong format.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="KuzuAdapter bug: get_model_independent_graph_data format mismatch")
async def test_get_graph_metrics(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    metrics = await adapter.get_graph_metrics()
    assert metrics["num_nodes"] == len(kg.nodes)
    assert metrics["num_edges"] == len(kg.edges)


# ---------------------------------------------------------------------------
# get_disconnected_nodes
# Known adapter bug: NOT EXISTS((n)-[]-()) syntax not supported in current Kuzu.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="KuzuAdapter bug: NOT EXISTS pattern syntax unsupported")
async def test_get_disconnected_nodes(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    # Before adding edges, all nodes should be disconnected
    disconnected = await adapter.get_disconnected_nodes()
    assert len(disconnected) == len(kg.nodes)

    # After adding edges, connected nodes should disappear from the list
    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    disconnected_after = await adapter.get_disconnected_nodes()
    assert len(disconnected_after) < len(disconnected)


# ---------------------------------------------------------------------------
# query (raw)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_query(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    assert result[0][0] == len(kg.nodes)


# ---------------------------------------------------------------------------
# is_empty after data + delete_graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_empty_after_add(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    assert await adapter.is_empty() is False


# ---------------------------------------------------------------------------
# get_triplets_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_triplets_batch(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [(e.source_node_id, e.target_node_id, e.relationship_name, {}) for e in kg.edges]
    await adapter.add_edges(edge_rows)

    triplets = await adapter.get_triplets_batch(offset=0, limit=10)
    assert len(triplets) == len(kg.edges)

    for t in triplets:
        assert "start_node" in t
        assert "relationship_properties" in t
        assert "end_node" in t


# ---------------------------------------------------------------------------
# close / subprocess lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_direct(kuzu_adapter):
    await kuzu_adapter.add_nodes(_load_demo_kg().nodes)
    await kuzu_adapter.close()


@pytest.mark.asyncio
async def test_close_subprocess(subprocess_adapter):
    await subprocess_adapter.add_nodes(_load_demo_kg().nodes)
    await subprocess_adapter.close()


@pytest.mark.asyncio
async def test_subprocess_calls_after_close_raise(subprocess_adapter):
    await subprocess_adapter.close()
    with pytest.raises(RuntimeError):
        await subprocess_adapter.query("MATCH (n) RETURN n")


# ---------------------------------------------------------------------------
# delete_graph drains in-flight queries before dropping native resources
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_graph_waits_for_in_flight_query(kuzu_adapter):
    """``delete_graph()`` must not tear down ``self.connection`` while an
    executor thread is mid-``blocking_query``. White-box: claim an
    ``open_connections`` slot directly to simulate an in-flight query,
    fire ``delete_graph`` on a separate task, and assert it parks until
    the slot is released.
    """
    import asyncio

    # Force ``open_connections > 0`` while ``delete_graph`` runs by
    # claiming a slot directly — equivalent to a query that's parked
    # inside ``run_in_executor`` but without needing to stub the executor.
    async with kuzu_adapter._connection_lock:
        kuzu_adapter.open_connections += 1
        kuzu_adapter._all_queries_drained.clear()

    delete_started = asyncio.Event()
    delete_finished = asyncio.Event()

    async def run_delete():
        delete_started.set()
        await kuzu_adapter.delete_graph()
        delete_finished.set()

    delete_task = asyncio.create_task(run_delete())

    # Give the delete a chance to enter ``_drain_in_flight_queries``.
    await delete_started.wait()
    await asyncio.sleep(0.05)

    # delete_graph must still be parked: connection NOT yet dropped.
    assert not delete_finished.is_set(), (
        "delete_graph completed while open_connections > 0 — drain is broken"
    )

    # Release the simulated in-flight query.
    kuzu_adapter.open_connections -= 1
    if kuzu_adapter.open_connections == 0:
        kuzu_adapter._all_queries_drained.set()

    # Now delete_graph should wake up and complete promptly.
    try:
        await asyncio.wait_for(delete_task, timeout=5.0)
    except asyncio.TimeoutError:  # pragma: no cover - regression signal
        delete_task.cancel()
        raise AssertionError("delete_graph did not complete after drain")


@pytest.mark.asyncio
async def test_close_does_not_block_event_loop_in_subprocess_mode(tmp_path):
    """``close()`` must not freeze the event loop while
    ``_drop_native_resources()`` issues its OP_CONN_CLOSE / OP_DB_CLOSE
    RPCs in subprocess mode, nor while ``session.shutdown()`` does its
    join/terminate/kill chain. Both should be offloaded via
    ``asyncio.to_thread`` so other coroutines on the same loop continue
    making progress.
    """
    import asyncio
    import time as _time

    a = create_graph_engine(
        graph_database_provider="kuzu",
        graph_file_path=str(tmp_path / "kuzu_subprocess_close"),
        graph_database_subprocess_enabled=True,
    )
    if hasattr(a, "initialize"):
        await a.initialize()

    # Replace ``_drop_native_resources`` and ``session.shutdown`` with slow
    # stubs so we can observe whether close() yields to the heartbeat.
    original_drop = a._drop_native_resources
    original_session_shutdown = a._session.shutdown

    # Make ``_drop_native_resources`` deliberately slow. ``session.shutdown``
    # is already offloaded by the surrounding fix in ``close()`` (item 12);
    # this test specifically pins the additional offload for the drop.
    def slow_drop():
        _time.sleep(0.3)
        original_drop()

    def slow_session_shutdown(*args, **kwargs):
        original_session_shutdown(*args, **kwargs)

    a._drop_native_resources = slow_drop
    a._session.shutdown = slow_session_shutdown

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    hb_task = asyncio.create_task(heartbeat())
    try:
        await a.close()
    finally:
        stop.set()
        await hb_task

    # 300 ms slow drop. With ``asyncio.to_thread``, the heartbeat ticks
    # every ~10 ms during the drop → expect ~30 ticks. Without the
    # offload, the drop blocks the loop and we see ≤ 2 ticks (one before
    # close starts, one after it returns). Threshold of 15 cleanly
    # discriminates while leaving slack for OS-scheduler jitter.
    assert ticks >= 15, (
        f"event loop appears blocked during close: only {ticks} heartbeats "
        f"during a 300ms slow drop — _drop_native_resources is not offloaded"
    )


@pytest.mark.asyncio
async def test_query_after_close_raises_clean_error(kuzu_adapter):
    """A ``query()`` issued after ``close()`` must surface a clean
    "adapter is closed" RuntimeError, not a low-level executor error
    (e.g. "cannot schedule new futures after shutdown") leaking through
    from the just-shut-down ThreadPoolExecutor. ``close()`` latches
    ``_permanently_closed`` at its start so the up-front check in
    ``query()`` catches the call before it reaches ``run_in_executor``.
    """
    await kuzu_adapter.close()

    with pytest.raises(RuntimeError, match="closed") as excinfo:
        await kuzu_adapter.query("MATCH (n) RETURN n")
    # Must not be the executor's own error message.
    assert "cannot schedule new futures" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_close_is_idempotent(kuzu_adapter):
    """``close()`` may be invoked multiple times (e.g. by both LRU
    eviction and ``__del__`` / explicit teardown). The second call must
    return cleanly without re-shutting-down anything."""
    await kuzu_adapter.close()
    # Second call should be a no-op — must not raise.
    await kuzu_adapter.close()


@pytest.mark.asyncio
async def test_query_does_not_block_event_loop_during_slow_redis_acquire(kuzu_adapter, monkeypatch):
    """``redis_lock.acquire_lock()`` and ``release_lock()`` are sync calls
    that do Redis I/O (default ``blocking_timeout=300s``). When invoked
    from ``query()`` on the event-loop thread they would freeze the
    loop. Both must be offloaded via ``asyncio.to_thread``.
    """
    import asyncio
    import time as _time

    # Reach through to the real adapter module — ``kuzu.adapter`` is now a
    # legacy import shim; the module-globals (``cache_config``,
    # ``ThreadPoolExecutor``, etc.) live on ``ladybug.adapter``.
    from cognee.infrastructure.databases.graph.ladybug import adapter as kuzu_adapter_mod

    class _SlowRedisLock:
        def __init__(self) -> None:
            self.acquired = False
            self.released = False

        def acquire_lock(self):
            _time.sleep(0.3)
            self.acquired = True
            return self

        def release_lock(self, lock=None) -> None:
            assert lock is self
            self.released = True

    stub = _SlowRedisLock()
    kuzu_adapter.redis_lock = stub
    # Force the shared-lock branch. ``cache_config`` is module-global on
    # the adapter module; mutating its ``shared_ladybug_lock`` flag is
    # enough to send ``query()`` down the redis-lock path.
    monkeypatch.setattr(kuzu_adapter_mod.cache_config, "shared_ladybug_lock", True)

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    hb_task = asyncio.create_task(heartbeat())
    try:
        await kuzu_adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    finally:
        stop.set()
        await hb_task

    assert stub.acquired, "redis lock acquire was not invoked"
    assert stub.released, "redis lock release was not invoked"
    # 300 ms slow acquire. With ``asyncio.to_thread``, heartbeat ticks
    # every ~10 ms during it → expect ~30 ticks. Without offload, the
    # loop is frozen for the full 300 ms and ticks would be ≤ 2.
    assert ticks >= 15, (
        f"event loop appears blocked during query: only {ticks} heartbeats "
        f"during a 300ms slow Redis acquire — acquire_lock is not offloaded"
    )


@pytest.mark.asyncio
async def test_query_racing_with_close_does_not_leak_executor_error(kuzu_adapter):
    """The race CodeRabbit flagged: ``close()`` shuts down the executor
    while a concurrent ``query()`` is between its closed-check and its
    ``run_in_executor`` call. With the sync ``_lifecycle_lock``,
    ``close()`` swaps ``self.executor = None`` under the lock before
    shutting down the *captured* executor, so query() either sees the
    closed flag (clean RuntimeError) or captured the executor while
    it was still live (run_in_executor completes against a stable ref).

    To exercise the window, we have many query tasks racing many close
    calls (close is idempotent). With the fix, every error surfaces as
    the clean "KuzuAdapter is closed" RuntimeError. Without it, some
    queries leak "cannot schedule new futures after shutdown".
    """
    import asyncio

    # Pre-warm so subsequent queries take the lazy-init shortcut.
    await kuzu_adapter.query("MATCH (n:Node) RETURN COUNT(n)")

    n_queries = 30
    errors_seen: list[BaseException] = []

    async def fire_query():
        try:
            await kuzu_adapter.query("MATCH (n:Node) RETURN COUNT(n)")
        except RuntimeError as exc:
            errors_seen.append(exc)
        except Exception as exc:  # pragma: no cover - any other type is a regression
            errors_seen.append(exc)

    async def fire_close():
        # Yield once so close races against in-flight queries rather
        # than running first.
        await asyncio.sleep(0)
        await kuzu_adapter.close()

    tasks = [asyncio.create_task(fire_query()) for _ in range(n_queries)]
    tasks.append(asyncio.create_task(fire_close()))
    await asyncio.gather(*tasks, return_exceptions=True)

    # Every error must be the clean "closed" message, never the
    # executor's leakage.
    for exc in errors_seen:
        msg = str(exc)
        assert "cannot schedule new futures" not in msg, (
            f"close/query race produced executor leakage: {exc!r}"
        )


@pytest.mark.asyncio
async def test_delete_graph_subprocess_recreates_schema(subprocess_adapter):
    """``delete_graph`` removes the on-disk store, so the reopened DB is
    empty and has no Node/EDGE tables. Without recreating the schema,
    the next graph query raises "table Node does not exist". Pins the
    ``_ensure_schema()`` call at the end of ``_rebuild_subprocess_proxies``.
    """
    kg = _load_demo_kg()
    await subprocess_adapter.add_nodes(kg.nodes)
    assert await subprocess_adapter.is_empty() is False

    await subprocess_adapter.delete_graph()

    # Adapter must still be usable: schema exists and DB is empty.
    assert await subprocess_adapter.is_empty() is True

    # Re-add to confirm the recreated schema actually accepts writes
    # (not just an empty SELECT against the catalog).
    await subprocess_adapter.add_nodes(kg.nodes)
    assert await subprocess_adapter.is_empty() is False
