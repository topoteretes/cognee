"""Lifecycle of edge-evidence rows: they follow the graph they describe.

Ingestion only appends to ``provenance_edge_evidence``; this module is the one
writer that removes or rewrites rows. A failed run's rollback keeps the
documents the run completed (``cognee/modules/cognify/rollback.py``), and
``keep_completed_items_evidence`` keeps their evidence readable with them.
Rows become garbage in two situations:

- the document is deleted (``forget()``, ``datasets.delete_data``,
  ``delete_dataset``) — swept from the relational data-entity delete;
- the document's memory is dropped (``forget(memory_only=True)``) — the edges
  the rows point at are gone, and the next cognify recaptures evidence under
  its own pipeline run.

Lookups already exclude rows whose ``Data`` row no longer exists (inner join),
so the sweep is hygiene against unbounded growth, not correctness. It never
raises: deletion must not fail because of a sidecar.
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .models import ProvenanceEdgeEvidence

logger = get_logger("provenance.cleanup")


def evidence_delete_statement(dataset_id: UUID, data_id: UUID | None = None):
    statement = delete(ProvenanceEdgeEvidence).where(
        ProvenanceEdgeEvidence.dataset_id == dataset_id
    )
    if data_id is not None:
        statement = statement.where(ProvenanceEdgeEvidence.data_id == data_id)
    return statement


async def delete_edge_evidence(dataset_id: UUID, data_id: UUID | None = None) -> int:
    """Remove the evidence rows of a dataset, or of one document in it."""
    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            result = await session.execute(evidence_delete_statement(dataset_id, data_id))
            await session.commit()
            return result.rowcount or 0
    except Exception as error:
        logger.warning(
            "Unable to remove edge evidence for dataset %s data %s: %s",
            dataset_id,
            data_id,
            error,
            exc_info=True,
        )
        return 0


async def keep_completed_items_evidence(
    dataset_id: UUID, pipeline_run_id: UUID, data_ids: set[UUID]
) -> int:
    """Keep the evidence of the documents a failed run completed readable.

    Lookups count a row only when its run has a completed record, so rows of an
    errored run are hidden. The rollback keeps the documents that run completed
    (stamped, and reported completed by the run), and the next cognify skips
    them, so nothing would ever write their evidence again. Their rows are
    detached from the run (``pipeline_run_id`` cleared, the form a row written
    outside a run has) and observed now, as a completed run's rows are observed
    at its completion, so conflict review sees them as new since its last
    review. Never raises: the sidecar must not fail the rollback.
    """
    if not data_ids:
        return 0
    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            result = await session.execute(
                update(ProvenanceEdgeEvidence)
                .where(
                    ProvenanceEdgeEvidence.dataset_id == dataset_id,
                    ProvenanceEdgeEvidence.pipeline_run_id == pipeline_run_id,
                    ProvenanceEdgeEvidence.data_id.in_(list(data_ids)),
                )
                .values(pipeline_run_id=None, created_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return result.rowcount or 0
    except Exception as error:
        logger.warning(
            "Unable to keep edge evidence of the completed items of run %s (dataset %s): %s",
            pipeline_run_id,
            dataset_id,
            error,
            exc_info=True,
        )
        return 0
