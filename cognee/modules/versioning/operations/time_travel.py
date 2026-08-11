from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.versioning.models.Checkpoint import Checkpoint
from cognee.modules.versioning.models.VersionEvent import VersionEvent


def _to_utc(dt: datetime) -> datetime:
    """Attach UTC to a timezone-naive datetime (SQLite returns naive datetimes)."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@with_async_session
async def get_nodes_at_time(
    dataset_id: UUID,
    target_time: datetime,
    *,
    session: AsyncSession,
) -> Dict:
    """Return the alive node-slug set for *dataset_id* at *target_time*.

    Algorithm (Approach 3 spec: nearest checkpoint + replay log tail):

    1. Find the latest Checkpoint for the dataset created at or before *target_time*.
    2. Seed the alive-node set from that checkpoint (empty set if none exists).
    3. Replay every ADD and FORGET event that falls between the checkpoint
       and *target_time*, in sequence order:
         - ADD   → union the node set
         - FORGET (not yet undone at *target_time*) → subtract the node set
    4. Return the resulting set of alive node slugs.

    The result can be used to filter a live graph/vector query to a past state
    without copying data.

    Args:
        dataset_id: Dataset to inspect.
        target_time: Point in time to reconstruct (timezone-aware recommended;
            naive datetimes are treated as UTC).

    Returns:
        Dict with keys:
        - ``alive_node_slugs``: sorted list of node UUID strings alive at
          *target_time*.
        - ``dataset_id``: stringified dataset UUID.
        - ``target_time``: ISO-8601 string.
        - ``checkpoint_id``: UUID of the checkpoint used as base, or ``None``.
        - ``checkpoint_time``: ISO-8601 string of checkpoint creation, or
          ``None``.
    """
    target_time = _to_utc(target_time)

    # ------------------------------------------------------------------
    # Step 1: find nearest checkpoint at or before target_time
    # ------------------------------------------------------------------
    cp_result = await session.execute(
        select(Checkpoint)
        .where(
            and_(
                Checkpoint.dataset_id == dataset_id,
                Checkpoint.created_at <= target_time,
            )
        )
        .order_by(Checkpoint.created_at.desc())
        .limit(1)
    )
    cp = cp_result.scalar_one_or_none()

    alive_nodes: set = set()
    if cp:
        cp_time = _to_utc(cp.created_at)
        alive_nodes = set(json.loads(cp.node_slugs or "[]"))
        replay_from = cp_time
    else:
        replay_from = datetime.min.replace(tzinfo=timezone.utc)

    # ------------------------------------------------------------------
    # Step 2: replay ADD/FORGET events from replay_from to target_time
    # ------------------------------------------------------------------
    events_result = await session.execute(
        select(VersionEvent)
        .where(
            and_(
                VersionEvent.dataset_id == dataset_id,
                VersionEvent.created_at > replay_from,
                VersionEvent.created_at <= target_time,
                VersionEvent.operation.in_(["ADD", "FORGET"]),
            )
        )
        .order_by(VersionEvent.sequence_number.asc())
    )

    for event in events_result.scalars().all():
        payload = json.loads(event.payload or "{}")
        node_slugs: set = set(payload.get("node_slugs", []))
        if event.operation == "ADD":
            alive_nodes |= node_slugs
        elif event.operation == "FORGET":
            # Only subtract if NOT undone before target_time
            undone = _to_utc(event.undone_at)
            if undone is None or undone > target_time:
                alive_nodes -= node_slugs

    return {
        "dataset_id": str(dataset_id),
        "target_time": target_time.isoformat(),
        "alive_node_slugs": sorted(alive_nodes),
        "checkpoint_id": str(cp.id) if cp else None,
        "checkpoint_time": _to_utc(cp.created_at).isoformat() if cp else None,
    }
