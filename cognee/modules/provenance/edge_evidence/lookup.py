"""Dataset-scoped reads from the append-only edge evidence sidecar."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, exists, func, or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.shared.logging_utils import get_logger

from .models import ProvenanceEdgeEvidence

logger = get_logger("provenance.lookup")
_EDGE_SOURCE_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class EdgeEvidenceRecord:
    edge_id: UUID
    data_id: UUID
    chunk_id: UUID
    chunk_index: int | None
    document_name: str | None
    external_metadata: dict | None = None
    observed_at: datetime | None = None


def _active_support(dataset_id: UUID, edge_ids: list[UUID]):
    completed_run_exists = exists(
        select(PipelineRun.id).where(
            PipelineRun.pipeline_run_id == ProvenanceEdgeEvidence.pipeline_run_id,
            PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        )
    )
    return (
        select(
            ProvenanceEdgeEvidence.edge_id,
            ProvenanceEdgeEvidence.data_id,
            ProvenanceEdgeEvidence.chunk_id,
            ProvenanceEdgeEvidence.chunk_index,
            func.min(ProvenanceEdgeEvidence.created_at).label("first_seen_at"),
            func.max(ProvenanceEdgeEvidence.created_at).label("observed_at"),
        )
        .where(
            ProvenanceEdgeEvidence.dataset_id == dataset_id,
            ProvenanceEdgeEvidence.edge_id.in_(edge_ids),
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

    distinct_support = _active_support(dataset_id, unique_edge_ids)
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
            Data,
            and_(
                Data.id == ranked_support.c.data_id,
                Data.dataset_id == dataset_id,
            ),
        )
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
        logger.debug("Unable to resolve graph edge evidence: %s", error, exc_info=True)
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


async def get_edge_sources(dataset_id: UUID, edge_ids: Iterable[UUID]) -> list[EdgeEvidenceRecord]:
    """Read every active support with its document date and latest observation.

    Review requires complete evidence, so unlike optional answer citations this
    read has no result limit and propagates storage errors. Batch edge IDs to
    stay below relational drivers' bind-parameter limits.
    """
    # Sorting before batching preserves the query's global edge/chunk ordering.
    edge_ids = sorted(set(edge_ids))
    if not edge_ids:
        return []
    records: dict[tuple[UUID, UUID, UUID], EdgeEvidenceRecord] = {}
    async with get_relational_engine().get_async_session() as session:
        for start in range(0, len(edge_ids), _EDGE_SOURCE_BATCH_SIZE):
            support = _active_support(dataset_id, edge_ids[start : start + _EDGE_SOURCE_BATCH_SIZE])
            statement = (
                select(
                    support.c.edge_id,
                    support.c.data_id,
                    support.c.chunk_id,
                    support.c.chunk_index,
                    Data.name,
                    Data.external_metadata,
                    support.c.observed_at,
                )
                .join(Data, and_(Data.id == support.c.data_id, Data.dataset_id == dataset_id))
                .order_by(support.c.edge_id, support.c.chunk_id)
            )
            for row in (await session.execute(statement)).all():
                record = EdgeEvidenceRecord(*row)
                key = (record.edge_id, record.data_id, record.chunk_id)
                previous = records.get(key)
                if previous is None or record.observed_at > previous.observed_at:
                    records[key] = record
    return list(records.values())


async def get_touched_entity_ids(dataset_id: UUID, since: datetime) -> set[str]:
    """Return candidate Entity endpoints observed by a completed write since the stamp."""
    evidence = ProvenanceEdgeEvidence
    completed_since = exists(
        select(PipelineRun.id).where(
            PipelineRun.pipeline_run_id == evidence.pipeline_run_id,
            PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            PipelineRun.created_at > since,
        )
    )
    statement = select(
        evidence.source_node_id,
        evidence.destination_node_id,
        evidence.evidence_kind,
    ).where(
        evidence.dataset_id == dataset_id,
        or_(
            evidence.evidence_kind == "extracted",
            and_(evidence.evidence_kind == "structural", evidence.relationship_name == "contains"),
        ),
        or_(
            completed_since,
            and_(evidence.pipeline_run_id.is_(None), evidence.created_at > since),
        ),
    )
    async with get_relational_engine().get_async_session() as session:
        rows = (await session.execute(statement)).all()
    touched = set()
    for source, target, kind in rows:
        touched.add(str(target))
        if kind == "extracted":
            touched.add(str(source))
    return touched
