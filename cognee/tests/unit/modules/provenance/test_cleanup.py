from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.provenance.edge_evidence import cleanup
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence


def _row(dataset_id, data_id):
    return ProvenanceEdgeEvidence(
        id=uuid4(),
        tenant_id=None,
        user_id=uuid4(),
        dataset_id=dataset_id,
        data_id=data_id,
        pipeline_run_id=uuid4(),
        chunk_id=uuid4(),
        chunk_index=0,
        edge_id=uuid4(),
        source_node_id=uuid4(),
        destination_node_id=uuid4(),
        relationship_name="knows",
        evidence_kind="extracted",
        created_at=datetime.now(timezone.utc),
    )


async def _seeded_engine(monkeypatch, rows):
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        await connection.run_sync(ProvenanceEdgeEvidence.__table__.create)
    async with session_factory() as session:
        session.add_all(rows)
        await session.commit()
    monkeypatch.setattr(
        cleanup,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=session_factory),
    )
    return sql_engine, session_factory


async def _count(session_factory, **filters):
    async with session_factory() as session:
        statement = select(func.count()).select_from(ProvenanceEdgeEvidence)
        for column, value in filters.items():
            statement = statement.where(getattr(ProvenanceEdgeEvidence, column) == value)
        return (await session.execute(statement)).scalar_one()


@pytest.mark.asyncio
async def test_delete_for_one_document_keeps_its_siblings(monkeypatch):
    dataset_id, other_dataset_id = uuid4(), uuid4()
    doomed, sibling, other = uuid4(), uuid4(), uuid4()
    sql_engine, sessions = await _seeded_engine(
        monkeypatch,
        [
            _row(dataset_id, doomed),
            _row(dataset_id, doomed),
            _row(dataset_id, sibling),
            _row(other_dataset_id, other),
        ],
    )

    assert await cleanup.delete_edge_evidence(dataset_id, doomed) == 2

    assert await _count(sessions, data_id=doomed) == 0
    assert await _count(sessions, data_id=sibling) == 1
    assert await _count(sessions, dataset_id=other_dataset_id) == 1
    await sql_engine.dispose()


@pytest.mark.asyncio
async def test_delete_for_dataset_sweeps_every_document_in_it_only(monkeypatch):
    dataset_id, other_dataset_id = uuid4(), uuid4()
    sql_engine, sessions = await _seeded_engine(
        monkeypatch,
        [
            _row(dataset_id, uuid4()),
            _row(dataset_id, uuid4()),
            _row(other_dataset_id, uuid4()),
        ],
    )

    assert await cleanup.delete_edge_evidence(dataset_id) == 2

    assert await _count(sessions, dataset_id=dataset_id) == 0
    assert await _count(sessions, dataset_id=other_dataset_id) == 1
    await sql_engine.dispose()


@pytest.mark.asyncio
async def test_sweep_failures_never_propagate(monkeypatch):
    """Deletion of a document must not fail because the sidecar is unavailable."""

    def _broken_engine():
        raise RuntimeError("relational database is down")

    monkeypatch.setattr(cleanup, "get_relational_engine", _broken_engine)

    assert await cleanup.delete_edge_evidence(uuid4(), uuid4()) == 0
