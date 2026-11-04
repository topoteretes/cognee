"""forget(everything=True) deletes sessions by the dataset rule, never the whole cache (SDK-1237).

Two users share one SQL session cache. User A forgets everything: A's own
session and the session attributed to a dataset A may delete go whole; user
B's other session, the cache's key/value rows and B's usage log stay. The
cache is never pruned.
"""

import importlib
from contextlib import ExitStack
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

    async def list_for_datasets(dataset_ids):
        return [pair for dataset_id in dataset_ids for pair in dataset_pairs.get(dataset_id, [])]

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
        patch("cognee.modules.session_lifecycle.list_sessions_for_datasets", list_for_datasets),
        # The package re-exports the function under its submodule's name, so a
        # dotted target fails on Python 3.10: patch the submodule object itself.
        patch.object(
            importlib.import_module("cognee.infrastructure.session.get_session_manager"),
            "get_session_manager",
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
            AsyncMock(return_value=set()),
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
    with patch.object(cache, "prune", prune), ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))

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
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))

    assert result["sessions_removed"] == 0
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_forget_everything_raises_before_deleting_datasets_when_a_session_delete_fails(
    cache,
):
    """A failed session delete keeps the datasets, so a retry finds the session again.

    The dataset half of the rule reaches another user's session only while the
    dataset exists; deleting the datasets first would orphan it on failure.
    """
    await _seed_session(cache, USER_B, "b_shared")
    manager = SessionManager(cache)
    patches = _patches(
        manager,
        session_pairs_by_owner={},
        dataset_pairs={DATASET_SHARED: [(USER_B, "b_shared")]},
        deletable_datasets=[SimpleNamespace(id=DATASET_SHARED)],
    )
    datasets_api = importlib.import_module("cognee.api.v1.datasets.datasets").datasets
    delete_session_module = importlib.import_module(
        "cognee.modules.session_lifecycle.delete_session"
    )
    failing_rows = AsyncMock(side_effect=RuntimeError("relational store down"))

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(
            patch.object(delete_session_module, "_delete_session_rows", failing_rows)
        )
        with pytest.raises(RuntimeError, match="relational store down"):
            await forget_module._forget_everything(SimpleNamespace(id=USER_A))
        datasets_api.delete_all.assert_not_awaited()

    # The retry still resolves the session through the surviving dataset.
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))
        datasets_api.delete_all.assert_awaited_once()

    assert result["status"] == "success"
    assert await _session_counts(cache, USER_B, "b_shared") == (0, 0, 0)


@pytest.mark.asyncio
async def test_forget_everything_deletes_the_callers_sessions_that_have_no_dashboard_row(cache):
    """session_records is written best-effort, and legacy sessions have no row at all:
    the caller's own sessions are read from the cache too, so none survives."""
    await _seed_session(cache, USER_A, "a_no_row")
    await _seed_session(cache, USER_B, "b_no_row")
    manager = SessionManager(cache)
    patches = _patches(
        manager,
        session_pairs_by_owner={},
        dataset_pairs={},
        deletable_datasets=[],
    )
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))

    assert result["sessions_removed"] == 1
    assert await _session_counts(cache, USER_A, "a_no_row") == (0, 0, 0)
    # Another user's session is reached only through its dataset row, never the cache listing.
    assert await _session_counts(cache, USER_B, "b_no_row") == (1, 1, 1)


@pytest.mark.asyncio
async def test_forget_everything_deletes_every_session_in_one_batch(cache):
    """Each store is detagged once for all sessions, not once per session."""
    for session_id in ("a1", "a2", "a3"):
        await _seed_session(cache, USER_A, session_id)
    await _seed_session(cache, USER_B, "b_shared")
    manager = SessionManager(cache)
    patches = _patches(
        manager,
        session_pairs_by_owner={USER_A: [(USER_A, "a1")]},
        dataset_pairs={DATASET_SHARED: [(USER_B, "b_shared")]},
        deletable_datasets=[SimpleNamespace(id=DATASET_SHARED)],
    )
    global_detag = AsyncMock()
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                global_detag,
            )
        )
        result = await forget_module._forget_everything(SimpleNamespace(id=USER_A))

    assert result["sessions_removed"] == 4
    global_detag.assert_awaited_once()
    detagged = set(global_detag.await_args.args[0])
    assert detagged == {
        (str(USER_A), "a1"),
        (str(USER_A), "a2"),
        (str(USER_A), "a3"),
        (str(USER_B), "b_shared"),
    }


@pytest.mark.asyncio
async def test_forget_everything_deletes_a_session_whose_owner_no_longer_exists(cache):
    """A deleted user leaves their session_records rows (no foreign key). Their session on a
    deletable dataset still goes whole, and the missing owner never aborts the forget."""
    from contextlib import asynccontextmanager

    from cognee.infrastructure.databases.exceptions import EntityNotFoundError

    await _seed_session(cache, USER_A, "a_own")
    await _seed_session(cache, USER_B, "b_shared")
    manager = SessionManager(cache)
    patches = _patches(
        manager,
        session_pairs_by_owner={USER_A: [(USER_A, "a_own")]},
        dataset_pairs={DATASET_SHARED: [(USER_B, "b_shared")]},
        deletable_datasets=[SimpleNamespace(id=DATASET_SHARED)],
    )
    user_a = SimpleNamespace(id=USER_A)

    async def get_user(user_id):
        if user_id == USER_B:
            raise EntityNotFoundError(message=f"Could not find user: {user_id}")
        return user_a

    entered = []

    @asynccontextmanager
    async def dataset_context(dataset_id, user_id):
        entered.append(dataset_id)
        yield

    dataset_detag = AsyncMock()
    module = "cognee.modules.session_lifecycle.delete_session"
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        for target, value in {
            "backend_access_control_enabled": lambda: True,
            "get_user": get_user,
            "get_all_user_permission_datasets": AsyncMock(return_value=[]),
            "_recorded_datasets": AsyncMock(return_value={(USER_B, "b_shared"): DATASET_SHARED}),
            "_existing_datasets": AsyncMock(return_value={DATASET_SHARED}),
            "set_database_global_context_variables": dataset_context,
            "delete_session_qa_vectors": dataset_detag,
        }.items():
            stack.enter_context(patch(f"{module}.{target}", value))
        result = await forget_module._forget_everything(user_a)

    assert result["status"] == "success"
    assert result["sessions_removed"] == 2
    assert await _session_counts(cache, USER_B, "b_shared") == (0, 0, 0)
    # The recorded dataset's store is still detagged for the deleted owner's session.
    assert entered == [DATASET_SHARED]
    assert set(dataset_detag.await_args.args[0]) == {(str(USER_B), "b_shared")}
