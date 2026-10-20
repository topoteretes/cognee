"""Explicit per-session delete through the public SDK surface (cognee.session.delete_session).

Runs over real stores: the filesystem and SQL session-cache adapters, and a
real SQLite relational database holding ``session_records`` and
``session_model_usage``. Only the Q&A vector hook is replaced, so the tests
can assert it was called for the deleted session and that its failure
propagates instead of being swallowed.
"""

import importlib
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord

delete_module = importlib.import_module("cognee.modules.session_lifecycle.delete_session")


@pytest_asyncio.fixture(params=["fs", "sql"])
async def stores(request):
    """A real session cache (fs or SQL) and a real SQLite relational DB, wired into the delete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        relational = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmpdir}/relational.db")
        async with relational.engine.begin() as connection:
            await connection.run_sync(
                SessionRecord.metadata.create_all,
                tables=[SessionRecord.__table__, SessionModelUsage.__table__],
            )

        if request.param == "fs":
            with patch(
                "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
                return_value={"data_root_directory": tmpdir},
            ):
                from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                    FSCacheAdapter,
                )

                adapter = FSCacheAdapter()
        else:
            from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import (
                SqlCacheAdapter,
            )

            adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmpdir}/cache.db")

        manager = SessionManager(cache_engine=adapter)
        vectors = AsyncMock()
        with (
            patch.object(delete_module, "get_session_manager", return_value=manager),
            patch.object(delete_module, "get_relational_engine", return_value=relational),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new=vectors,
            ),
        ):
            yield SimpleNamespace(manager=manager, relational=relational, vectors=vectors)

        if request.param == "fs":
            adapter.cache.close()
        else:
            await adapter.close()
        await relational.engine.dispose()


async def _seed_session(stores, user_id, session_id):
    """One session's full footprint: a Q&A turn, a trace step, a context row, and its DB rows."""
    uid = str(user_id)
    await stores.manager._cache.create_qa_entry(
        uid, session_id, question="q", context="c", answer="a", qa_id=str(uuid4())
    )
    await stores.manager._cache.append_agent_trace_step(
        user_id=uid, session_id=session_id, trace_id=str(uuid4()), origin_function="f", status="ok"
    )
    await stores.manager._cache.create_session_context_entry(
        uid, session_id, {"id": str(uuid4()), "kind": "context", "text": "lesson"}
    )
    async with stores.relational.get_async_session() as db:
        db.add(SessionRecord(session_id=session_id, user_id=user_id))
        db.add(SessionModelUsage(session_id=session_id, user_id=user_id, model="gpt"))
        await db.commit()


async def _footprint(stores, user_id, session_id) -> dict:
    uid = str(user_id)
    async with stores.relational.get_async_session() as db:
        records = (
            await db.execute(
                select(SessionRecord).where(
                    SessionRecord.user_id == user_id, SessionRecord.session_id == session_id
                )
            )
        ).all()
        usage = (
            await db.execute(
                select(SessionModelUsage).where(
                    SessionModelUsage.user_id == user_id,
                    SessionModelUsage.session_id == session_id,
                )
            )
        ).all()
    return {
        "qa": len(await stores.manager.get_session(user_id=uid, session_id=session_id) or []),
        "traces": len(
            await stores.manager.get_agent_trace_session(user_id=uid, session_id=session_id)
        ),
        "context": len(
            await stores.manager.get_session_context_entries(user_id=uid, session_id=session_id)
        ),
        "records": len(records),
        "usage": len(usage),
    }


FULL = {"qa": 1, "traces": 1, "context": 1, "records": 1, "usage": 1}
EMPTY = {"qa": 0, "traces": 0, "context": 0, "records": 0, "usage": 0}


@pytest.mark.asyncio
async def test_delete_session_removes_every_trace_of_the_session(stores):
    owner, stranger = uuid4(), uuid4()
    await _seed_session(stores, owner, "s1")
    await _seed_session(stores, owner, "s2")
    await _seed_session(stores, stranger, "s1")
    assert await _footprint(stores, owner, "s1") == FULL

    assert await cognee.session.delete_session("s1", user=SimpleNamespace(id=owner)) is True

    assert await _footprint(stores, owner, "s1") == EMPTY
    stores.vectors.assert_awaited_once_with(user_id=str(owner), session_id="s1")
    # The owner's other session and another user's session of the same id stay.
    assert await _footprint(stores, owner, "s2") == FULL
    assert await _footprint(stores, stranger, "s1") == FULL


@pytest.mark.asyncio
async def test_delete_session_reports_a_missing_session(stores):
    owner = uuid4()
    await _seed_session(stores, owner, "s1")
    user = SimpleNamespace(id=owner)

    assert await cognee.session.delete_session("s1", user=user) is True
    assert await cognee.session.delete_session("s1", user=user) is False
    assert await cognee.session.delete_session("never-existed", user=user) is False


@pytest.mark.asyncio
async def test_delete_session_removes_a_dashboard_row_without_cache_content(stores):
    """A session whose cache content expired still has its dashboard row; deleting it counts."""
    owner = uuid4()
    async with stores.relational.get_async_session() as db:
        db.add(SessionRecord(session_id="expired", user_id=owner))
        await db.commit()

    assert await cognee.session.delete_session("expired", user=SimpleNamespace(id=owner)) is True
    assert (await _footprint(stores, owner, "expired"))["records"] == 0


@pytest.mark.asyncio
async def test_delete_session_propagates_a_vector_cleanup_failure(stores):
    """Q&A vectors that could not be removed would keep answering recall: the delete raises."""
    owner = uuid4()
    await _seed_session(stores, owner, "s1")
    stores.vectors.side_effect = RuntimeError("vector store unreachable")

    with pytest.raises(RuntimeError, match="vector store unreachable"):
        await cognee.session.delete_session("s1", user=SimpleNamespace(id=owner))


@pytest.mark.asyncio
async def test_delete_session_raises_when_session_memory_is_disabled():
    with (
        patch.object(
            delete_module, "get_session_manager", return_value=SessionManager(cache_engine=None)
        ),
        pytest.raises(CogneeValidationError, match="CACHING=false"),
    ):
        await cognee.session.delete_session("s1", user=SimpleNamespace(id=uuid4()))


@pytest.mark.asyncio
async def test_delete_session_requires_a_session_id():
    with pytest.raises(CogneeValidationError, match="explicit session_id"):
        await cognee.session.delete_session("", user=SimpleNamespace(id=uuid4()))
