"""Lifecycle of edge-evidence rows: they follow the graph they describe.

Ingestion only appends to ``provenance_edge_evidence``; this module is the one
writer that removes rows. Rows become garbage in two situations:

- the document is deleted (``forget()``, ``datasets.delete_data``,
  ``delete_dataset``) — swept from the relational data-entity delete;
- the document's memory is dropped (``forget(memory_only=True)``) — the edges
  the rows point at are gone, and the next cognify recaptures evidence under
  its own pipeline run.

Lookups already exclude rows whose ``Data`` row no longer exists (inner join),
so the sweep is hygiene against unbounded growth, not correctness. It never
raises: deletion must not fail because of a sidecar.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .models import ProvenanceEdgeEvidence

logger = get_logger("provenance.cleanup")


def evidence_delete_statement(dataset_id: UUID, data_id: Optional[UUID] = None):
    statement = delete(ProvenanceEdgeEvidence).where(
        ProvenanceEdgeEvidence.dataset_id == dataset_id
    )
    if data_id is not None:
        statement = statement.where(ProvenanceEdgeEvidence.data_id == data_id)
    return statement


async def delete_edge_evidence(dataset_id: UUID, data_id: Optional[UUID] = None) -> int:
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
        )
        return 0
