from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.legacy.GraphRelationshipLedger import GraphRelationshipLedger
from cognee.modules.versioning.models.VersionEvent import DEFAULT_RETENTION_DAYS, VersionEvent

BATCH_SIZE = 1000


class UndoForgetResult:
    """Result returned by :func:`undo_forget`."""

    def __init__(
        self,
        event_id: UUID,
        node_slugs: List[str],
        edge_slugs: List[str],
        ledger_rows_restored: int,
        datapoints_for_reingest: List[dict],
    ) -> None:
        self.event_id = event_id
        self.node_slugs = node_slugs
        self.edge_slugs = edge_slugs
        self.ledger_rows_restored = ledger_rows_restored
        # Deserialized DataPoint dicts from matching ADD events; callers can
        # re-add these via cognee.add() to fully restore graph/vector data.
        self.datapoints_for_reingest = datapoints_for_reingest

    def to_dict(self) -> Dict:
        return {
            "event_id": str(self.event_id),
            "node_slugs": self.node_slugs,
            "edge_slugs": self.edge_slugs,
            "ledger_rows_restored": self.ledger_rows_restored,
            "datapoints_for_reingest_count": len(self.datapoints_for_reingest),
            "note": (
                "Ledger soft-deletes cleared. "
                "datapoints_for_reingest contains the original DataPoint payloads "
                "from the corresponding ADD events (if captured). "
                "Call cognee.add() with this data to fully restore graph/vector stores."
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
    """Reverse the most recent FORGET event for a dataset.

    Steps
    -----
    1. Locate the most recent unresolved FORGET ``VersionEvent`` for the
       given ``dataset_id`` (filtered further by ``data_id`` / ``event_id``
       when supplied).
    2. Check the retention window — raise ``ValueError`` if ``expires_at`` has
       passed (data may have been compacted).
    3. Parse the payload to retrieve affected ``node_slugs`` and ``edge_slugs``.
    4. Clear ``deleted_at`` on matching ``GraphRelationshipLedger`` rows.
    5. Look up matching ADD events for those node IDs to recover the original
       ``DataPoint`` JSON — this is the re-ingest payload.
    6. Mark the FORGET event as ``undone_at = now()``.

    Since graph/vector data is hard-deleted, full restoration requires calling
    ``cognee.add()`` with the returned ``datapoints_for_reingest`` list.
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

    # Retention window check (normalise to UTC — SQLite returns naive datetimes)
    now = datetime.now(timezone.utc)
    expires = event.expires_at
    if expires is not None and expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires is not None and expires < now:
        raise ValueError(
            f"FORGET event {event.id} has expired at {event.expires_at.isoformat()} "
            f"(now={now.isoformat()}). Data may have been compacted — undo not possible."
        )

    payload = json.loads(event.payload or "{}")
    node_slugs: List[str] = payload.get("node_slugs", [])
    edge_slugs: List[str] = payload.get("edge_slugs", [])

    # Clear deleted_at on ledger entries for affected source nodes
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

    # Retrieve DataPoint JSON from matching ADD events so callers can re-ingest
    datapoints_for_reingest: List[dict] = []
    if node_slugs:
        add_events_query = (
            select(VersionEvent)
            .where(
                and_(
                    VersionEvent.dataset_id == dataset_id,
                    VersionEvent.operation == "ADD",
                    VersionEvent.data_id == data_id if data_id else True,
                )
            )
            .order_by(VersionEvent.sequence_number.asc())
        )
        add_result = await session.execute(add_events_query)
        add_events = add_result.scalars().all()

        node_slug_set = set(node_slugs)
        for add_event in add_events:
            add_payload = json.loads(add_event.payload or "{}")
            add_node_ids = set(add_payload.get("node_slugs", []))
            # If this ADD event covers any of the forgotten nodes, include
            # its DataPoint snapshots for re-ingestion
            if add_node_ids & node_slug_set:
                for dp_json in add_payload.get("datapoints", []):
                    try:
                        datapoints_for_reingest.append(json.loads(dp_json))
                    except (json.JSONDecodeError, TypeError):
                        pass

    # Mark the FORGET event as undone
    event.undone_at = now
    session.add(event)
    await session.flush()

    # Append a RESTORE event so the audit log is complete
    restore_seq_result = await session.execute(
        select(func.coalesce(func.max(VersionEvent.sequence_number), 0)).where(
            VersionEvent.dataset_id == dataset_id
        )
    )
    restore_seq = (restore_seq_result.scalar_one() or 0) + 1
    restore_event = VersionEvent(
        operation="RESTORE",
        dataset_id=dataset_id,
        data_id=event.data_id,
        user_id=event.user_id,
        run_id=event.run_id,
        sequence_number=restore_seq,
        created_at=now,
        expires_at=now + timedelta(days=DEFAULT_RETENTION_DAYS),
        payload=json.dumps({
            "node_slugs": node_slugs,
            "edge_slugs": edge_slugs,
            "datapoints": [],
            "restores_event_id": str(event.id),
        }),
    )
    session.add(restore_event)
    await session.flush()

    return UndoForgetResult(
        event_id=event.id,
        node_slugs=node_slugs,
        edge_slugs=edge_slugs,
        ledger_rows_restored=rows_restored,
        datapoints_for_reingest=datapoints_for_reingest,
    )
