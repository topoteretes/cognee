"""Real SQL transactions: concurrent writers, lost ACKs, conflicts and crash recovery."""

import asyncio
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, update

from cognee.infrastructure.databases.cache.sql.capture import CaptureConflictError, store_capture
from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.cache.sql.tables import (
    cache_capture_receipts,
    cache_qa_entries,
    cache_trace_entries,
)


@pytest_asyncio.fixture(params=["sqlite", "postgres"])
async def adapter(tmp_path, request):
    if request.param == "postgres":
        uri = os.environ.get("COGNEE_TEST_CAPTURE_POSTGRES_URL")
        if not uri:
            pytest.skip("Set COGNEE_TEST_CAPTURE_POSTGRES_URL to a disposable Postgres database")
    else:
        uri = f"sqlite+aiosqlite:///{tmp_path}/cache.db"
    instance = SqlCacheAdapter(uri)
    await instance._ensure_initialized()
    await instance.prune()
    yield instance
    await instance.close()


def request(
    adapter,
    event_id,
    *,
    answer="answer",
    user="user",
    dataset="dataset",
    tenant="tenant",
    session="session",
    kind="qa",
    builds=None,
    vectors=None,
    fail_index=False,
):
    entry = {"type": kind, "question": "question", "answer": answer}

    async def build(entry_id):
        if builds is not None:
            builds.append(entry_id)
        if kind == "qa":
            return adapter._build_qa_entry_dump("question", "", answer, entry_id, None, None)
        return adapter._build_agent_trace_entry_dump(
            trace_id=entry_id,
            origin_function="Bash",
            status="success",
        )

    async def index(entry_id, payload):
        if vectors is not None:
            vectors.append(entry_id)
        if fail_index:
            raise OSError("lost vector acknowledgement")

    return store_capture(
        adapter,
        tenant_id=tenant,
        user_id=user,
        dataset_id=dataset,
        dataset_name="name",
        session_id=session,
        client_event_id=event_id,
        entry=entry,
        build_payload=build,
        index_qa=index,
    )


async def count(adapter, table):
    async with adapter.engine.connect() as connection:
        return (await connection.execute(select(func.count()).select_from(table))).scalar_one()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["qa", "trace"])
async def test_concurrent_process_connections_admit_once(adapter, kind):
    peers = [adapter, *(SqlCacheAdapter(adapter.db_uri) for _ in range(3))]
    builds, vectors, event_id = [], [], str(uuid4())
    try:
        for peer in peers:
            await peer._ensure_initialized()
        results = await asyncio.gather(
            *[
                request(peer, event_id, kind=kind, builds=builds, vectors=vectors)
                for peer in peers
                for _ in range(3)
            ]
        )
        assert all(receipt == results[0] for receipt in results)
        assert len(builds) == 1
        assert len(vectors) == (1 if kind == "qa" else 0)
        assert await count(adapter, cache_capture_receipts) == 1
        assert await count(adapter, cache_qa_entries if kind == "qa" else cache_trace_entries) == 1
    finally:
        for peer in peers[1:]:
            await peer.close()


@pytest.mark.asyncio
async def test_commit_before_ack_reopen_returns_stable_receipt(adapter):
    event_id, vectors = str(uuid4()), []
    committed = await request(adapter, event_id, vectors=vectors)
    # The client did not receive this response. Reopen as a different server worker.
    peer = SqlCacheAdapter(adapter.db_uri)
    try:
        assert await request(peer, event_id, vectors=vectors) == committed
        assert vectors == [committed["entry_id"]]
        assert await count(peer, cache_qa_entries) == 1
    finally:
        await peer.close()


@pytest.mark.asyncio
async def test_changed_payload_conflicts_without_mutation(adapter):
    event_id = str(uuid4())
    first = await request(adapter, event_id)
    with pytest.raises(CaptureConflictError):
        await request(adapter, event_id, answer="different")
    assert await request(adapter, event_id) == first
    assert await count(adapter, cache_qa_entries) == 1


@pytest.mark.asyncio
async def test_separate_identical_events_and_scopes_remain_separate(adapter):
    event_id = str(uuid4())
    results = [await request(adapter, event_id)]
    results.append(await request(adapter, str(uuid4())))
    for scope in (
        {"user": "other"},
        {"dataset": "other"},
        {"tenant": "other"},
        {"session": "other"},
    ):
        results.append(await request(adapter, event_id, **scope))
    assert len({r["entry_id"] for r in results}) == 6
    assert await count(adapter, cache_capture_receipts) == 6
    assert await count(adapter, cache_qa_entries) == 6


@pytest.mark.asyncio
async def test_embedding_failure_reuses_committed_entry_and_vector_id(adapter):
    event_id, builds, vectors = str(uuid4()), [], []
    with pytest.raises(OSError):
        await request(adapter, event_id, builds=builds, vectors=vectors, fail_index=True)
    receipt = await request(adapter, event_id, builds=builds, vectors=vectors)
    assert len(builds) == 1 and vectors == [receipt["entry_id"], receipt["entry_id"]]
    assert await count(adapter, cache_qa_entries) == 1
    assert await request(adapter, event_id, builds=builds, vectors=vectors) == receipt
    assert len(vectors) == 2


@pytest.mark.asyncio
async def test_rollback_does_not_leave_a_false_receipt(adapter):
    async def broken(_):
        raise OSError("payload construction failed")

    with pytest.raises(OSError):
        await store_capture(
            adapter,
            tenant_id="tenant",
            user_id="user",
            dataset_id="dataset",
            dataset_name="name",
            session_id="session",
            client_event_id="event",
            entry={"type": "qa"},
            build_payload=broken,
            index_qa=None,
        )
    assert await count(adapter, cache_capture_receipts) == 0
    assert await count(adapter, cache_qa_entries) == 0


@pytest.mark.asyncio
async def test_retry_after_forget_does_not_resurrect_entry(adapter):
    event_id, vectors = str(uuid4()), []
    receipt = await request(adapter, event_id, vectors=vectors)
    await adapter.delete_session("user", "session")
    assert await request(adapter, event_id, vectors=vectors) == receipt
    assert await count(adapter, cache_qa_entries) == 0
    assert len(vectors) == 1


@pytest.mark.asyncio
async def test_expired_worker_lease_recovers_without_a_second_entry(adapter):
    event_id, vectors = str(uuid4()), []
    receipt = await request(adapter, event_id, vectors=vectors)
    # Simulate a dead worker that indexed but never persisted completion.
    async with adapter.sessionmaker() as session, session.begin():
        await session.execute(
            update(cache_capture_receipts).values(
                indexed=False, index_owner="dead-worker", index_lease_until=1
            )
        )
    assert await request(adapter, event_id, vectors=vectors) == receipt
    assert vectors == [receipt["entry_id"], receipt["entry_id"]]
    assert await count(adapter, cache_qa_entries) == 1


@pytest.mark.asyncio
async def test_embedding_does_not_block_an_independent_writer(adapter):
    started, release = asyncio.Event(), asyncio.Event()

    async def index(*_):
        started.set()
        await release.wait()

    async def build(entry_id):
        return adapter._build_qa_entry_dump("q", "", "a", entry_id, None, None)

    task = asyncio.create_task(
        store_capture(
            adapter,
            tenant_id="tenant",
            user_id="user",
            dataset_id="dataset",
            dataset_name="name",
            session_id="session",
            client_event_id=str(uuid4()),
            entry={"type": "qa", "question": "q", "answer": "a"},
            build_payload=build,
            index_qa=index,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        other = await asyncio.wait_for(request(adapter, str(uuid4())), timeout=2)
        assert other["entry_id"]
    finally:
        release.set()
        await task
    assert await count(adapter, cache_qa_entries) == 2


@pytest.mark.asyncio
async def test_prune_clears_receipts_and_allows_fresh_admission(adapter):
    event_id, builds = str(uuid4()), []
    await request(adapter, event_id, builds=builds)
    await adapter.prune()
    assert await count(adapter, cache_capture_receipts) == 0
    assert await count(adapter, cache_qa_entries) == 0
    await request(adapter, event_id, builds=builds)
    assert len(builds) == 2


@pytest.mark.asyncio
async def test_retention_sweep_keeps_recent_receipts_and_active_leases(adapter):
    import time
    from datetime import timedelta

    for event in ("old", "recent", "leased"):
        await request(adapter, event)
    async with adapter.sessionmaker() as session, session.begin():
        await session.execute(
            update(cache_capture_receipts)
            .where(cache_capture_receipts.c.client_event_id.in_(["old", "leased"]))
            .values(created_at=adapter._now() - timedelta(days=91))
        )
        await session.execute(
            update(cache_capture_receipts)
            .where(cache_capture_receipts.c.client_event_id == "leased")
            .values(index_lease_until=time.time() + 120)
        )
    adapter._last_purge = 0
    await adapter._maybe_purge_expired()
    async with adapter.sessionmaker() as session:
        assert set(
            (await session.execute(select(cache_capture_receipts.c.client_event_id))).scalars()
        ) == {"recent", "leased"}


@pytest.mark.asyncio
async def test_account_cleanup_removes_only_that_users_receipts(adapter):
    await request(adapter, "one", user="deleted")
    await request(adapter, "two", user="other")
    await adapter.delete_capture_receipts_for_user("deleted")
    async with adapter.sessionmaker() as session:
        assert list(
            (await session.execute(select(cache_capture_receipts.c.user_id))).scalars()
        ) == ["other"]


@pytest.mark.asyncio
async def test_receipt_lookup_uses_original_selector_and_authenticated_scope(adapter):
    from cognee.infrastructure.databases.cache.sql.capture import find_capture

    receipt = await request(adapter, "event")
    args = {
        "tenant_id": "tenant",
        "user_id": "user",
        "dataset_id": None,
        "dataset_name": "name",
        "session_id": "session",
        "client_event_id": "event",
        "entry": {"type": "qa", "question": "question", "answer": "answer"},
    }
    assert (await find_capture(adapter, **args))["receipt"] == receipt
    assert (
        await find_capture(adapter, **{**args, "dataset_id": "dataset", "dataset_name": "ignored"})
    )["receipt"] == receipt
    for key in ("tenant_id", "user_id", "session_id", "dataset_id", "client_event_id"):
        assert await find_capture(adapter, **{**args, key: "other"}) is None
    with pytest.raises(CaptureConflictError):
        await find_capture(adapter, **{**args, "entry": {"type": "qa", "answer": "changed"}})
