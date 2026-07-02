from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.versioning.models.VersionEvent import VersionEvent, DEFAULT_RETENTION_DAYS

BATCH_SIZE = 1000


@with_async_session
async def log_version_event(
    operation: str,
    dataset_id: UUID,
    *,
    data_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    run_id: Optional[UUID] = None,
    node_slugs: Optional[List[str]] = None,
    edge_slugs: Optional[List[str]] = None,
    datapoint_snapshots: Optional[List[str]] = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    session: AsyncSession,
) -> VersionEvent:
    """Append one VersionEvent to the event log and return it.

    ``operation`` must be one of: ``"ADD"``, ``"COGNIFY"``, ``"FORGET"``.

    ``node_slugs`` and ``edge_slugs`` are recorded for FORGET events so that
    :func:`undo_forget` can locate the affected nodes.

    ``datapoint_snapshots`` is a list of ``DataPoint.to_json()`` strings captured
    from the ADD pipeline, stored in the payload so undo_forget can signal
    re-ingestion with the exact original data.

    The ``sequence_number`` is computed as ``MAX(sequence_number) + 1`` for the
    dataset within the same session, so callers don't need to manage it.
    ``expires_at`` defaults to ``now + retention_days``.
    """
    # Compute per-dataset sequence number atomically
    seq_result = await session.execute(
        select(func.coalesce(func.max(VersionEvent.sequence_number), 0)).where(
            VersionEvent.dataset_id == dataset_id
        )
    )
    next_seq: int = (seq_result.scalar_one() or 0) + 1

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=retention_days)

    payload = json.dumps(
        {
            "node_slugs": node_slugs or [],
            "edge_slugs": edge_slugs or [],
            "datapoints": datapoint_snapshots or [],
        }
    )

    event = VersionEvent(
        operation=operation,
        dataset_id=dataset_id,
        data_id=data_id,
        user_id=user_id,
        run_id=run_id,
        sequence_number=next_seq,
        created_at=now,
        expires_at=expires_at,
        payload=payload,
    )
    session.add(event)
    await session.flush()
    await session.refresh(event)
    return event
