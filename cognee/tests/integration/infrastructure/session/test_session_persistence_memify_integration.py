"""
Integration tests: session persistence memify pipeline + SessionManager.

Runs with FsCache and in-memory Redis (parametrized). Uses a module-scoped
event loop so cognee graph/vector engines are not reused across different loops.
"""
import asyncio
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.low_level import setup
from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.modules.users.methods import get_default_user


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so all tests and the env fixture share one loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class _InMemoryRedisList:
    """Minimal in-memory Redis list emulation for Redis-backed session tests."""

    def __init__(self):
        self.data: dict[str, list[str]] = {}

    async def rpush(self, key: str, *vals: str):
        self.data.setdefault(key, []).extend(vals)

    async def lrange(self, key: str, start: int, end: int):
        lst = self.data.get(key, [])
        s = start if start >= 0 else len(lst) + start
        e = (end + 1) if end >= 0 else len(lst) + end + 1
        return lst[s:e]

    async def lindex(self, key: str, idx: int):
        lst = self.data.get(key, [])
        return lst[idx] if -len(lst) <= idx < len(lst) else None

    async def lset(self, key: str, idx: int, val: str):
        self.data[key][idx] = val

    async def delete(self, key: str):
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        pass

    async def flushdb(self):
        self.data.clear()


@pytest_asyncio.fixture(scope="module")
async def session_persistence_env(event_loop):
    """Clean cognee env with one dataset (add + cognify); shared by all tests in this module."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_path = str(base_dir / ".cognee_system/test_session_persistence_memify")
    data_path = str(base_dir / ".data_storage/test_session_persistence_memify")

    cognee.config.system_root_directory(system_path)
    cognee.config.data_root_directory(data_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    dataset_name = "session_persist_integration"
    await cognee.add("Cognee builds knowledge graphs from text.", dataset_name=dataset_name)
    await cognee.cognify(datasets=[dataset_name])

    yield dataset_name

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.fixture(params=["fs", "redis"])
def session_manager_with_qa(request):
    """
    SessionManager backed by either FsCache or in-memory Redis.
    Tests run twice (once per backend).
    """
    backend = request.param
    if backend == "fs":
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
                return_value={"data_root_directory": tmpdir},
            ):
                from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                    FSCacheAdapter,
                )

                adapter = FSCacheAdapter()
                sm = SessionManager(cache_engine=adapter)
                yield sm, adapter
                adapter.cache.close()
    else:
        store = _InMemoryRedisList()
        patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
        with (
            patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
            patch(f"{patch_mod}.aioredis.Redis", return_value=store),
        ):
            from cognee.infrastructure.databases.cache.redis.RedisAdapter import (
                RedisAdapter,
            )

            adapter = RedisAdapter(host="localhost", port=6379)
            sm = SessionManager(cache_engine=adapter)
            yield sm, adapter


@pytest.mark.asyncio
async def test_persist_sessions_in_knowledge_graph_uses_session_manager(
    session_persistence_env,
    session_manager_with_qa,
):
    """
    Session persistence memify pipeline reads from SessionManager and writes to the graph.

    - Populate SessionManager with Q&A for a session.
    - Patch extract_user_sessions to use this SessionManager.
    - Run persist_sessions_in_knowledge_graph_pipeline.
    - Assert session content appears in the graph (DocumentChunk count increases).
    """
    dataset_name = session_persistence_env
    session_manager, _adapter = session_manager_with_qa

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_qa(
        user_id=user_id,
        question="What does Cognee do?",
        context="Intro to Cognee",
        answer="Cognee builds knowledge graphs from text.",
        session_id="integration_test_session",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=["integration_test_session"],
            dataset=dataset_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()

    document_chunk_count = 0
    for _node_id, props in nodes:
        t = props.get("type")
        if isinstance(t, dict) and t.get("DocumentChunk"):
            document_chunk_count += 1
        elif t == "DocumentChunk":
            document_chunk_count += 1

    assert document_chunk_count >= 2, (
        "Expected at least 2 DocumentChunks (1 from initial add+cognify, 1 from session). "
        f"Got {document_chunk_count} DocumentChunk nodes, total nodes: {len(nodes)}"
    )


@pytest.mark.asyncio
async def test_persist_sessions_multiple_sessions_via_session_manager(
    session_persistence_env,
    session_manager_with_qa,
):
    """Persist two sessions from SessionManager; both are cognified into the graph."""
    dataset_name = session_persistence_env
    session_manager, _adapter = session_manager_with_qa

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_qa(
        user_id=user_id,
        question="First question?",
        context="C1",
        answer="First answer.",
        session_id="session_a",
    )
    await session_manager.add_qa(
        user_id=user_id,
        question="Second question?",
        context="C2",
        answer="Second answer.",
        session_id="session_b",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=["session_a", "session_b"],
            dataset=dataset_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()

    assert len(nodes) >= 3, (
        "Expected at least 3 nodes (1 from initial doc + 2 from sessions). "
        f"Got {len(nodes)} nodes."
    )
