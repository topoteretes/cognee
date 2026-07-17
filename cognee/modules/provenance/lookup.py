"""Dataset-scoped reads from the append-only edge evidence sidecar."""

from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from sqlalchemy import and_, exists, func, or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, DatasetData
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.shared.logging_utils import get_logger

from .models import ProvenanceEdgeEvidence

logger = get_logger("provenance.lookup")


@dataclass(frozen=True, slots=True)
class EdgeEvidenceRecord:
    edge_id: UUID
    data_id: UUID
    chunk_id: UUID
    chunk_index: int | None
    document_name: str | None


async def get_edge_evidence_records(
    edge_ids: Iterable[UUID],
    dataset_id: UUID,
    *,
    per_edge_limit: int = 5,
    total_limit: int = 50,
) -> list[EdgeEvidenceRecord]:
    """Return active source chunks for graph edges in one indexed query.

    Append-only observations are considered active when they have no run id or
    their pipeline run has a completed terminal record. Failed/rolled-back runs
    therefore need no delete-side mutation.
    """
    unique_edge_ids = list(dict.fromkeys(edge_ids))
    if not unique_edge_ids or per_edge_limit <= 0 or total_limit <= 0:
        return []

    completed_run_exists = exists(
        select(PipelineRun.id).where(
            PipelineRun.pipeline_run_id == ProvenanceEdgeEvidence.pipeline_run_id,
            PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        )
    )
    distinct_support = (
        select(
            ProvenanceEdgeEvidence.edge_id,
            ProvenanceEdgeEvidence.data_id,
            ProvenanceEdgeEvidence.chunk_id,
            ProvenanceEdgeEvidence.chunk_index,
            func.min(ProvenanceEdgeEvidence.created_at).label("first_seen_at"),
        )
        .where(
            ProvenanceEdgeEvidence.dataset_id == dataset_id,
            ProvenanceEdgeEvidence.edge_id.in_(unique_edge_ids),
            or_(
                ProvenanceEdgeEvidence.pipeline_run_id.is_(None),
                completed_run_exists,
            ),
        )
        .group_by(
            ProvenanceEdgeEvidence.edge_id,
            ProvenanceEdgeEvidence.data_id,
            ProvenanceEdgeEvidence.chunk_id,
            ProvenanceEdgeEvidence.chunk_index,
        )
        .subquery()
    )
    ranked_support = select(
        distinct_support,
        func.row_number()
        .over(
            partition_by=distinct_support.c.edge_id,
            order_by=distinct_support.c.first_seen_at,
        )
        .label("support_rank"),
    ).subquery()
    statement = (
        select(
            ranked_support.c.edge_id,
            ranked_support.c.data_id,
            ranked_support.c.chunk_id,
            ranked_support.c.chunk_index,
            Data.name,
        )
        .join(
            DatasetData,
            and_(
                DatasetData.dataset_id == dataset_id,
                DatasetData.data_id == ranked_support.c.data_id,
            ),
        )
        .join(Data, Data.id == ranked_support.c.data_id)
        .where(ranked_support.c.support_rank <= per_edge_limit)
        .order_by(ranked_support.c.edge_id, ranked_support.c.support_rank)
        .limit(total_limit)
    )

    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            rows = (await session.execute(statement)).all()
    except Exception as error:
        # References are optional and existing databases may briefly serve
        # traffic before their migration completes. Never fail the answer.
        logger.debug("Unable to resolve graph edge evidence: %s", error)
        return []

    counts: dict[UUID, int] = {}
    seen: set[tuple[UUID, UUID, UUID]] = set()
    records: list[EdgeEvidenceRecord] = []
    for edge_id, data_id, chunk_id, chunk_index, document_name in rows:
        key = (edge_id, data_id, chunk_id)
        if key in seen or counts.get(edge_id, 0) >= per_edge_limit:
            continue
        seen.add(key)
        counts[edge_id] = counts.get(edge_id, 0) + 1
        records.append(
            EdgeEvidenceRecord(
                edge_id=edge_id,
                data_id=data_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                document_name=document_name,
            )
        )
        if len(records) >= total_limit:
            break
    return records
