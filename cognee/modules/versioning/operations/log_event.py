from __future__ import annotations

import json
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.versioning.models.VersionEvent import VersionEvent

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
    session: AsyncSession,
) -> VersionEvent:
    """Append one VersionEvent to the event log and return it.

    ``operation`` must be one of: ``"ADD"``, ``"COGNIFY"``, ``"FORGET"``.
    ``node_slugs`` and ``edge_slugs`` are recorded in the event payload so
    that FORGET events can later be undone via :func:`undo_forget`.
    """
    payload = json.dumps(
        {
            "node_slugs": node_slugs or [],
            "edge_slugs": edge_slugs or [],
        }
    )

    event = VersionEvent(
        operation=operation,
        dataset_id=dataset_id,
        data_id=data_id,
        user_id=user_id,
        run_id=run_id,
        payload=payload,
    )
    session.add(event)
    await session.flush()
    await session.refresh(event)
    return event
