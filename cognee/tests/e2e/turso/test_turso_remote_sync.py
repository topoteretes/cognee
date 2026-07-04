"""Live integration tests for Turso remote/embedded-replica sync.

Skipped unless `TEST_TURSO_REMOTE_URL` and `TEST_TURSO_AUTH_TOKEN` are set.
Configure them to point at a real (ideally disposable/test) Turso database —
these tests write to and delete from it.
"""

import os
import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

TURSO_REMOTE_URL = os.getenv("TEST_TURSO_REMOTE_URL")
TURSO_AUTH_TOKEN = os.getenv("TEST_TURSO_AUTH_TOKEN")

pytestmark = pytest.mark.skipif(
    not (TURSO_REMOTE_URL and TURSO_AUTH_TOKEN),
    reason="TEST_TURSO_REMOTE_URL/TEST_TURSO_AUTH_TOKEN not set; skipping live Turso sync tests",
)


async def _make_adapter(local_path: str) -> TursoAdapter:
    adapter = TursoAdapter(
        connection_string=f"sqlite+aioturso:///{local_path}",
        remote_url=TURSO_REMOTE_URL,
        auth_token=TURSO_AUTH_TOKEN,
        sync_interval_seconds=0,  # always pull immediately in tests
    )
    await adapter.initialize()
    return adapter


@pytest_asyncio.fixture
async def replica_a(tmp_path):
    adapter = await _make_adapter(str(tmp_path / "replica_a.db"))
    yield adapter
    await adapter.delete_graph()
    await adapter.close()


@pytest.mark.asyncio
async def test_write_visible_from_second_replica(replica_a, tmp_path):
    await replica_a.add_node("n1", {"name": "Alice", "type": "Person"})
    await replica_a.add_node("n2", {"name": "Bob", "type": "Person"})
    await replica_a.add_edge("n1", "n2", "KNOWS")

    replica_b = await _make_adapter(str(tmp_path / "replica_b.db"))
    try:
        await replica_b._sync_pull()
        node = await replica_b.get_node("n1")
        assert node is not None
        assert node["name"] == "Alice"
        assert await replica_b.has_edge("n1", "n2", "KNOWS")
    finally:
        await replica_b.close()


@pytest.mark.asyncio
async def test_fresh_replica_bootstraps_from_remote(replica_a, tmp_path):
    await replica_a.add_node("n1", {"name": "Alice", "type": "Person"})

    replica_c = await _make_adapter(str(tmp_path / "replica_c.db"))
    try:
        node = await replica_c.get_node("n1")
        assert node is not None
        assert node["name"] == "Alice"
    finally:
        await replica_c.close()


@pytest.mark.asyncio
async def test_delete_propagates_through_push_and_pull(replica_a, tmp_path):
    await replica_a.add_node("n1", {"name": "Alice", "type": "Person"})

    replica_b = await _make_adapter(str(tmp_path / "replica_b.db"))
    try:
        await replica_b._sync_pull()
        assert await replica_b.get_node("n1") is not None

        await replica_a.delete_node("n1")

        await replica_b._sync_pull()
        assert await replica_b.get_node("n1") is None
    finally:
        await replica_b.close()
