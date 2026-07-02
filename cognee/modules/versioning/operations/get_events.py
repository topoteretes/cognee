from __future__ import annotations

import json
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.versioning.models.VersionEvent import VersionEvent


@with_async_session
async def get_event_log(
    dataset_id: UUID,
    *,
    data_id: Optional[UUID] = None,
    operation: Optional[str] = None,
    limit: int = 100,
    session: AsyncSession,
) -> List[dict]:
    """Return the event log for *dataset_id*, newest first.

    Args:
        dataset_id: Filter to this dataset.
        data_id: Further narrow to a specific data item (optional).
        operation: One of ``"ADD"``, ``"COGNIFY"``, ``"FORGET"`` (optional).
        limit: Maximum number of events to return (default 100).

    Returns:
        List of dicts with keys ``id``, ``operation``, ``dataset_id``,
        ``data_id``, ``created_at``, ``undone_at``, ``node_slugs``,
        ``edge_slugs``.
    """
    query = select(VersionEvent).where(VersionEvent.dataset_id == dataset_id)
    if data_id is not None:
        query = query.where(VersionEvent.data_id == data_id)
    if operation is not None:
        query = query.where(VersionEvent.operation == operation)

    query = query.order_by(VersionEvent.created_at.desc()).limit(limit)
    result = await session.execute(query)
    rows = result.scalars().all()

    events = []
    for row in rows:
        payload = json.loads(row.payload or "{}")
        events.append(
            {
                "id": str(row.id),
                "operation": row.operation,
                "dataset_id": str(row.dataset_id),
                "data_id": str(row.data_id) if row.data_id else None,
                "user_id": str(row.user_id) if row.user_id else None,
                "run_id": str(row.run_id) if row.run_id else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "undone_at": row.undone_at.isoformat() if row.undone_at else None,
                "node_slugs": payload.get("node_slugs", []),
                "edge_slugs": payload.get("edge_slugs", []),
            }
        )
    return events
