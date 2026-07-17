from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.provenance import persistence
from cognee.modules.provenance.buffer import ProvenanceBatch
from cognee.modules.provenance.models import ProvenanceEdgeEvidence


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_args):
        return False


def _row():
    return {
        "id": uuid4(),
        "tenant_id": uuid4(),
        "user_id": uuid4(),
        "dataset_id": uuid4(),
        "data_id": uuid4(),
        "pipeline_run_id": uuid4(),
        "chunk_id": uuid4(),
        "chunk_index": 0,
        "edge_id": uuid4(),
        "source_node_id": uuid4(),
        "destination_node_id": uuid4(),
        "relationship_name": "knows",
        "evidence_kind": "extracted",
        "source_task": "extract_graph_from_data",
        "confidence": None,
        "created_at": datetime.now(timezone.utc),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["sqlite", "postgresql"])
async def test_persistence_uses_one_statement_for_large_batches(monkeypatch, dialect):
    session = SimpleNamespace(
        connection=AsyncMock(return_value=SimpleNamespace(dialect=SimpleNamespace(name=dialect))),
        execute=AsyncMock(),
        commit=AsyncMock(),
    )
    engine = SimpleNamespace(get_async_session=lambda: _AsyncContext(session))
    monkeypatch.setattr(persistence, "get_relational_engine", lambda: engine)
    batch = ProvenanceBatch(evidence_rows=tuple(_row() for _ in range(1_000)))

    assert await persistence.persist_provenance_batch(batch) == 1_000

    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_batch_performs_no_database_access(monkeypatch):
    get_engine = AsyncMock()
    monkeypatch.setattr(persistence, "get_relational_engine", get_engine)

    assert await persistence.persist_provenance_batch(ProvenanceBatch(())) == 0
    get_engine.assert_not_called()


@pytest.mark.asyncio
async def test_sqlite_batch_is_retry_idempotent(monkeypatch):
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        await connection.run_sync(ProvenanceEdgeEvidence.__table__.create)

    monkeypatch.setattr(
        persistence,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=session_factory),
    )
    row = _row()
    batch = ProvenanceBatch((row,))

    await persistence.persist_provenance_batch(batch)
    await persistence.persist_provenance_batch(batch)

    async with session_factory() as session:
        count = (
            await session.execute(select(func.count()).select_from(ProvenanceEdgeEvidence))
        ).scalar_one()
        assert count == 1

    await sql_engine.dispose()
