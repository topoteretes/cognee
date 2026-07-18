from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data, Dataset, DatasetData
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.provenance import lookup
from cognee.modules.provenance.models import ProvenanceEdgeEvidence


@pytest.mark.asyncio
async def test_lookup_returns_only_evidence_from_completed_runs(monkeypatch):
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        for table in (
            Data.__table__,
            Dataset.__table__,
            DatasetData.__table__,
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
        session.add(Data(id=data_id, name="report.txt"))
        session.add(Dataset(id=dataset_id, name="reports"))
        session.add(DatasetData(dataset_id=dataset_id, data_id=data_id))
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
