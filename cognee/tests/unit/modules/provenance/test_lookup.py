import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.provenance.edge_evidence import lookup
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence


@pytest.mark.asyncio
async def test_lookup_returns_only_evidence_from_completed_runs(monkeypatch):
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        for table in (
            Data.__table__,
            Dataset.__table__,
            PipelineRun.__table__,
            ProvenanceEdgeEvidence.__table__,
        ):
            await connection.run_sync(
                lambda sync_connection, table=table: table.create(sync_connection)
            )

    monkeypatch.setattr(
        lookup,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=session_factory),
    )
    dataset_id = uuid4()
    data_id = uuid4()
    completed_run_id = uuid4()
    failed_run_id = uuid4()
    completed_edge_id = uuid4()
    failed_edge_id = uuid4()
    now = datetime.now(timezone.utc)

    async with session_factory() as session:
        session.add(Data(id=data_id, name="report.txt", dataset_id=dataset_id))
        session.add(Dataset(id=dataset_id, name="reports"))
        session.add(
            PipelineRun(
                pipeline_run_id=completed_run_id,
                status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            )
        )
        for edge_id, run_id in (
            (completed_edge_id, completed_run_id),
            (failed_edge_id, failed_run_id),
        ):
            session.add(
                ProvenanceEdgeEvidence(
                    id=uuid4(),
                    tenant_id=None,
                    user_id=uuid4(),
                    dataset_id=dataset_id,
                    data_id=data_id,
                    pipeline_run_id=run_id,
                    chunk_id=uuid4(),
                    chunk_index=2,
                    edge_id=edge_id,
                    source_node_id=uuid4(),
                    destination_node_id=uuid4(),
                    relationship_name="knows",
                    evidence_kind="extracted",
                    created_at=now,
                )
            )
        await session.commit()

    records = await lookup.get_edge_evidence_records(
        [completed_edge_id, failed_edge_id], dataset_id
    )

    assert len(records) == 1
    assert records[0].edge_id == completed_edge_id
    assert records[0].data_id == data_id
    assert records[0].document_name == "report.txt"
    await sql_engine.dispose()


@pytest.mark.asyncio
async def test_review_sources_and_touched_entities_use_complete_active_evidence(monkeypatch):
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        for table in (Data.__table__, PipelineRun.__table__, ProvenanceEdgeEvidence.__table__):
            await connection.run_sync(lambda sync, table=table: table.create(sync))
    monkeypatch.setattr(
        lookup, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    dataset_id, data_id, edge_id, chunk_id = [uuid4() for _ in range(4)]
    run_id, failed_run_id = uuid4(), uuid4()
    source, target, structural_target = uuid4(), uuid4(), uuid4()
    before = datetime(2025, 1, 1, tzinfo=timezone.utc)
    since = before + timedelta(days=1)
    after = before + timedelta(days=2)

    def observation(**overrides):
        properties = {
            "id": uuid4(),
            "user_id": uuid4(),
            "dataset_id": dataset_id,
            "data_id": data_id,
            "pipeline_run_id": run_id,
            "chunk_id": chunk_id,
            "chunk_index": 1,
            "edge_id": edge_id,
            "source_node_id": source,
            "destination_node_id": target,
            "relationship_name": "has_ceo",
            "evidence_kind": "extracted",
            "created_at": before,
        }
        return ProvenanceEdgeEvidence(**(properties | overrides))

    async with sessions() as session:
        session.add(
            Data(
                id=data_id,
                dataset_id=dataset_id,
                name="report",
                external_metadata={"effective_date": "2024-03-02"},
            )
        )
        # A row observed before the watermark still counts when its run completes later.
        session.add(
            PipelineRun(
                pipeline_run_id=run_id,
                status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                created_at=after,
            )
        )
        session.add_all(
            [
                observation(),
                observation(created_at=after, chunk_index=2),
                observation(
                    pipeline_run_id=failed_run_id,
                    chunk_id=uuid4(),
                    source_node_id=uuid4(),
                    destination_node_id=uuid4(),
                    created_at=after,
                ),
                observation(
                    dataset_id=uuid4(),
                    source_node_id=uuid4(),
                    destination_node_id=uuid4(),
                    created_at=after,
                ),
                observation(
                    evidence_kind="structural",
                    relationship_name="contains",
                    source_node_id=chunk_id,
                    destination_node_id=structural_target,
                ),
                observation(
                    evidence_kind="structural",
                    relationship_name="is_part_of",
                    source_node_id=uuid4(),
                    destination_node_id=uuid4(),
                ),
                observation(
                    pipeline_run_id=None,
                    created_at=before,
                    source_node_id=uuid4(),
                    destination_node_id=uuid4(),
                ),
            ]
        )
        no_run_source, no_run_target = uuid4(), uuid4()
        session.add(
            observation(
                pipeline_run_id=None,
                created_at=after,
                source_node_id=no_run_source,
                destination_node_id=no_run_target,
            )
        )
        # The sources API has no retrieval-style total/per-edge limit.
        session.add_all([observation(chunk_id=uuid4()) for _ in range(60)])
        # No Data row means this support cannot be cited, even after completion.
        session.add(observation(data_id=uuid4(), chunk_id=uuid4()))
        await session.commit()

    records = await lookup.get_edge_sources(dataset_id, [edge_id])
    assert len(records) == 61
    original = next(record for record in records if record.chunk_id == chunk_id)
    assert original.observed_at.replace(tzinfo=timezone.utc) == after
    assert original.external_metadata == {"effective_date": "2024-03-02"}
    assert original.document_name == "report"
    assert await lookup.get_touched_entity_ids(dataset_id, since) == {
        str(value) for value in (source, target, structural_target, no_run_source, no_run_target)
    }
    assert 0 < len(await lookup.get_edge_evidence_records([edge_id], dataset_id)) <= 5
    await sql_engine.dispose()


@pytest.mark.asyncio
async def test_review_source_read_propagates_storage_failures(monkeypatch):
    def unavailable():
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(lookup, "get_relational_engine", unavailable)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await lookup.get_edge_sources(uuid4(), [uuid4()])


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(sqlite3.Connection, "setlimit"), reason="requires Python 3.11+")
async def test_review_sources_exceed_sqlite_bind_limit_without_losing_evidence(monkeypatch):
    class LimitedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)

    sql_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", connect_args={"factory": LimitedConnection}
    )
    sessions = async_sessionmaker(sql_engine, expire_on_commit=False)
    monkeypatch.setattr(
        lookup, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    dataset_id, data_id, chunk_id = uuid4(), uuid4(), uuid4()
    edge_ids = [UUID(int=value) for value in range(1, 1004)]
    before = datetime(2025, 1, 1, tzinfo=timezone.utc)
    after = before + timedelta(days=1)
    try:
        async with sql_engine.begin() as connection:
            for table in (Data.__table__, PipelineRun.__table__, ProvenanceEdgeEvidence.__table__):
                await connection.run_sync(lambda sync, table=table: table.create(sync))
        async with sessions() as session:
            session.add(Data(id=data_id, dataset_id=dataset_id, name="report"))
            for edge_id, observed_at, chunk_index in [
                *((edge_id, before, 1) for edge_id in edge_ids),
                (edge_ids[0], after, 2),
            ]:
                session.add(
                    ProvenanceEdgeEvidence(
                        id=uuid4(),
                        user_id=uuid4(),
                        dataset_id=dataset_id,
                        data_id=data_id,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        edge_id=edge_id,
                        source_node_id=uuid4(),
                        destination_node_id=uuid4(),
                        relationship_name="knows",
                        evidence_kind="extracted",
                        created_at=observed_at,
                    )
                )
            await session.commit()

        # Prove this driver rejects the unbatched read, rather than only asserting call counts.
        async with sessions() as session:
            with pytest.raises(OperationalError, match="too many SQL variables"):
                await session.execute(select(lookup._active_support(dataset_id, edge_ids)))

        records = await lookup.get_edge_sources(dataset_id, [*reversed(edge_ids), edge_ids[0]])
        assert [record.edge_id for record in records] == edge_ids
        assert records[0].observed_at.replace(tzinfo=timezone.utc) == after
        assert records[0].chunk_index == 2
        assert all(record.document_name == "report" for record in records)
    finally:
        await sql_engine.dispose()


@pytest.mark.asyncio
async def test_lookup_excludes_evidence_whose_document_was_deleted(monkeypatch):
    """Rows have no foreign key to Data; the read side must not cite a gone document."""
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        for table in (
            Data.__table__,
            Dataset.__table__,
            PipelineRun.__table__,
            ProvenanceEdgeEvidence.__table__,
        ):
            await connection.run_sync(
                lambda sync_connection, table=table: table.create(sync_connection)
            )

    monkeypatch.setattr(
        lookup,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=session_factory),
    )
    dataset_id, data_id, edge_id = uuid4(), uuid4(), uuid4()

    async with session_factory() as session:
        session.add(Dataset(id=dataset_id, name="reports"))
        session.add(Data(id=data_id, name="report.txt", dataset_id=dataset_id))
        session.add(
            ProvenanceEdgeEvidence(
                id=uuid4(),
                tenant_id=None,
                user_id=uuid4(),
                dataset_id=dataset_id,
                data_id=data_id,
                pipeline_run_id=None,
                chunk_id=uuid4(),
                chunk_index=0,
                edge_id=edge_id,
                source_node_id=uuid4(),
                destination_node_id=uuid4(),
                relationship_name="knows",
                evidence_kind="extracted",
                created_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    assert len(await lookup.get_edge_evidence_records([edge_id], dataset_id)) == 1

    async with session_factory() as session:
        await session.delete(await session.get(Data, data_id))
        await session.commit()

    assert await lookup.get_edge_evidence_records([edge_id], dataset_id) == []
    await sql_engine.dispose()
