from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.legacy.GraphRelationshipLedger import GraphRelationshipLedger
from cognee.modules.versioning.models.VersionEvent import VersionEvent

BATCH_SIZE = 1000


class UndoForgetResult:
    """Result returned by :func:`undo_forget`."""

    def __init__(
        self,
        event_id: UUID,
        node_slugs: List[str],
        edge_slugs: List[str],
        ledger_rows_restored: int,
    ) -> None:
        self.event_id = event_id
        self.node_slugs = node_slugs
        self.edge_slugs = edge_slugs
        self.ledger_rows_restored = ledger_rows_restored

    def to_dict(self) -> Dict:
        return {
            "event_id": str(self.event_id),
            "node_slugs": self.node_slugs,
            "edge_slugs": self.edge_slugs,
            "ledger_rows_restored": self.ledger_rows_restored,
            "note": (
                "Ledger soft-deletes cleared.  Graph/vector data was hard-deleted and "
                "must be restored by re-running cognee.add() + cognee.cognify() for the "
                "affected data files."
            ),
        }


@with_async_session
async def undo_forget(
    dataset_id: UUID,
    *,
    data_id: Optional[UUID] = None,
    event_id: Optional[UUID] = None,
    session: AsyncSession,
) -> UndoForgetResult:
    """Reverse a FORGET event by clearing ledger soft-deletes.

    Looks up the most recent unresolved FORGET VersionEvent for the given
    ``dataset_id`` (and optionally ``data_id`` or ``event_id``), then:

    1. Parses the payload to retrieve the affected node/edge slugs.
    2. Clears ``deleted_at`` on matching ``GraphRelationshipLedger`` rows.
    3. Marks the VersionEvent as ``undone_at = now()``.

    Since graph/vector data is hard-deleted, this operation restores only
    the ledger audit trail.  Full data restoration requires re-ingestion via
    ``cognee.add()`` + ``cognee.cognify()``.  The returned
    :class:`UndoForgetResult` lists the affected slugs so callers can do so.
    """
    query = select(VersionEvent).where(
        and_(
            VersionEvent.dataset_id == dataset_id,
            VersionEvent.operation == "FORGET",
            VersionEvent.undone_at.is_(None),
        )
    )
    if data_id is not None:
        query = query.where(VersionEvent.data_id == data_id)
    if event_id is not None:
        query = query.where(VersionEvent.id == event_id)

    query = query.order_by(VersionEvent.created_at.desc()).limit(1)
    result = await session.execute(query)
    event = result.scalar_one_or_none()

    if event is None:
        raise ValueError(
            f"No unresolved FORGET event found for dataset_id={dataset_id}"
            + (f", data_id={data_id}" if data_id else "")
            + (f", event_id={event_id}" if event_id else "")
        )

    payload = json.loads(event.payload or "{}")
    node_slugs: List[str] = payload.get("node_slugs", [])
    edge_slugs: List[str] = payload.get("edge_slugs", [])

    # Clear deleted_at on ledger entries whose source or destination slug is
    # in the affected set, restoring them to the "alive" state.
    rows_restored = 0
    if node_slugs:
        try:
            node_uuids = [UUID(s) for s in node_slugs]
        except (ValueError, AttributeError):
            node_uuids = []

        for start in range(0, len(node_uuids), BATCH_SIZE):
            batch = node_uuids[start : start + BATCH_SIZE]
            stmt = (
                update(GraphRelationshipLedger)
                .where(
                    and_(
                        GraphRelationshipLedger.deleted_at.isnot(None),
                        GraphRelationshipLedger.source_node_id.in_(batch),
                    )
                )
                .values(deleted_at=None)
                .execution_options(synchronize_session="fetch")
            )
            res = await session.execute(stmt)
            rows_restored += res.rowcount

    # Mark the event as undone
    event.undone_at = datetime.now(timezone.utc)
    session.add(event)
    await session.flush()

    return UndoForgetResult(
        event_id=event.id,
        node_slugs=node_slugs,
        edge_slugs=edge_slugs,
        ledger_rows_restored=rows_restored,
    )
