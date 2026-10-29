"""forget(everything=True) deletes sessions by the dataset rule, never the whole cache (SDK-1237).

Two users share one SQL session cache. User A forgets everything: A's own
session and the session attributed to a dataset A may delete go whole; user
B's other session, the cache's key/value rows and B's usage log stay. The
cache is never pruned.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.session.session_manager import SessionManager

forget_module = importlib.import_module("cognee.api.v1.forget.forget")

USER_A = uuid4()
USER_B = uuid4()
DATASET_SHARED = uuid4()  # owned by B, A holds a delete grant
DATASET_PRIVATE = uuid4()  # B's own


@pytest_asyncio.fixture
async def cache(tmp_path):
    adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmp_path}/cache.db")
    yield adapter
    await adapter.close()


async def _seed_session(cache, user_id, session_id):
    await cache.create_qa_entry(
        str(user_id), session_id, question="q", context="c", answer="a", qa_id=f"{session_id}_qa"
    )
    await cache.append_agent_trace_step(
        str(user_id), session_id, trace_id=f"{session_id}_t", origin_function="f", status="ok"
    )
    await cache.create_session_context_entry(
        str(user_id), session_id, {"id": f"{session_id}_ctx", "kind": "context"}
    )


async def _session_counts(cache, user_id, session_id) -> tuple[int, int, int]:
    manager = SessionManager(cache)
    qa = await manager.get_session(user_id=str(user_id), session_id=session_id)
    traces = await cache.get_agent_trace_count(str(user_id), session_id)
    context = await cache.get_session_context_entries(str(user_id), session_id)
    return len(qa), traces, len(context)


def _patches(manager, session_pairs_by_owner, dataset_pairs, deletable_datasets):
    """forget's data paths replaced; the session deletion runs for real over ``manager``."""
    datasets_api = importlib.import_module("cognee.api.v1.datasets.datasets").datasets

    async def list_for_user(user_id):
        return session_pairs_by_owner.get(user_id, [])

    async def list_for_dataset(dataset_id):
        return dataset_pairs.get(dataset_id, [])

    return [
        patch.object(datasets_api, "list_datasets", AsyncMock(return_value=deletable_datasets)),
        patch.object(datasets_api, "delete_all", AsyncMock()),
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            AsyncMock(return_value=deletable_datasets),
        ),
        patch(
            "cognee.modules.search.operations.delete_user_search_history",
            AsyncMock(return_value={"queries_deleted": 0, "results_deleted": 0}),
        ),
        patch("cognee.modules.session_lifecycle.list_sessions_for_user", list_for_user),
        patch("cognee.modules.session_lifecycle.list_sessions_for_dataset", list_for_dataset),
        patch(
            "cognee.infrastructure.session.get_session_manager.get_session_manager",
            lambda *_a, **_k: manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.get_session_manager",
            lambda *_a, **_k: manager,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.backend_access_control_enabled",
            lambda: False,
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session._delete_session_rows",
            AsyncMock(return_value=0),
        ),
        patch(
            "cognee.modules.session_lifecycle.delete_session.delete_session_qa_vectors",
            AsyncMock(),
        ),
        patch(
            "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
            AsyncMock(),
        ),
    ]


@pytest.mark.asyncio
async def test_forget_everything_deletes_only_the_callers_and_deletable_datasets_sessions(cache):
    await _seed_session(cache, USER_A, "a_own")
    await _seed_session(cache, USER_B, "b_shared")
    await _seed_session(cache, USER_B, "b_private")
    await cache.set_value("lock:improve", "held")
    await cache.log_usage(str(USER_B), {"endpoint": "POST /v1/recall"})

    manager = SessionManager(cache)
    prune = AsyncMock()
    patches = _patches(
        manager,
        session_pairs_by_owner={USER_A: [(USER_A, "a_own")]},
        dataset_pairs={DATASET_SHARED: [(USER_B, "b_shared")]},
        deletable_datasets=[SimpleNamespace(id=DATASET_SHARED)],
    )
    with patch.object(cache, "prune", prune):
        for p in patches:
            p.start()
        try:
            result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))
        finally:
            for p in reversed(patches):
                p.stop()

    assert result["sessions_removed"] == 2
    assert result["datasets_removed"] == 1
    prune.assert_not_awaited()

    assert await _session_counts(cache, USER_A, "a_own") == (0, 0, 0)
    assert await _session_counts(cache, USER_B, "b_shared") == (0, 0, 0)
    assert await _session_counts(cache, USER_B, "b_private") == (1, 1, 1)
    assert await cache.get_value("lock:improve") == "held"
    assert len(await cache.get_usage_logs(str(USER_B))) == 1


@pytest.mark.asyncio
async def test_forget_everything_skips_sessions_when_session_memory_is_off(cache):
    """CACHING=false: no cache to delete from, and nothing to raise about."""
    manager = SessionManager(None)
    patches = _patches(
        manager,
        session_pairs_by_owner={USER_A: [(USER_A, "a_own")]},
        dataset_pairs={},
        deletable_datasets=[],
    )
    for p in patches:
        p.start()
    try:
        result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))
    finally:
        for p in reversed(patches):
            p.stop()

    assert result["sessions_removed"] == 0
    assert result["status"] == "success"
