from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.legacy.GraphRelationshipLedger import GraphRelationshipLedger
from cognee.modules.versioning.models.Checkpoint import Checkpoint

BATCH_SIZE = 1000


@with_async_session
async def create_checkpoint(
    dataset_id: UUID,
    *,
    user_id: Optional[UUID] = None,
    label: Optional[str] = None,
    session: AsyncSession,
) -> Checkpoint:
    """Snapshot the alive node/edge set for *dataset_id* into a Checkpoint row.

    The alive set is derived from ``GraphRelationshipLedger`` entries where
    ``deleted_at IS NULL`` — exactly the same source the existing delete
    pipeline uses.  This keeps the checkpoint consistent with the ledger
    without introducing any new bookkeeping.

    Returns the newly created :class:`Checkpoint` instance.
    """
    alive_query = select(GraphRelationshipLedger).where(
        GraphRelationshipLedger.deleted_at.is_(None)
    )
    result = await session.execute(alive_query)
    alive_rows = result.scalars().all()

    node_slugs = list({str(row.source_node_id) for row in alive_rows})
    destination_slugs = list({str(row.destination_node_id) for row in alive_rows})
    all_node_slugs = list(set(node_slugs + destination_slugs))

    edge_slugs = list({row.node_label for row in alive_rows if row.node_label})

    checkpoint = Checkpoint(
        dataset_id=dataset_id,
        user_id=user_id,
        label=label,
        node_slugs=json.dumps(all_node_slugs),
        edge_slugs=json.dumps(edge_slugs),
    )
    session.add(checkpoint)
    await session.flush()
    await session.refresh(checkpoint)
    return checkpoint
