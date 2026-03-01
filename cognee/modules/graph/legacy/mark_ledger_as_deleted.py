from uuid import UUID
from datetime import datetime, timezone
from typing import List

from sqlalchemy import and_, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from .GraphRelationshipLedger import GraphRelationshipLedger


@with_async_session
async def mark_ledger_nodes_as_deleted(node_slugs: List[UUID], session: AsyncSession) -> None:
    """Mark legacy ledger entries as deleted for the given node IDs.

    When non-legacy nodes are deleted from graph/vector DBs, their corresponding
    ledger entries (if any) should be marked as deleted so they are excluded from
    future legacy checks.
    """
    if not node_slugs:
        return

    stmt = (
        update(GraphRelationshipLedger)
        .where(
            and_(
                GraphRelationshipLedger.deleted_at.is_(None),
                GraphRelationshipLedger.source_node_id.in_(node_slugs),
                GraphRelationshipLedger.source_node_id
                == GraphRelationshipLedger.destination_node_id,
            )
        )
        .values(deleted_at=datetime.now(timezone.utc))
    )
    await session.execute(stmt)
    await session.commit()


@with_async_session
async def mark_ledger_edges_as_deleted(
    edge_relationship_names: List[str], session: AsyncSession
) -> None:
    """Mark legacy ledger edge entries as deleted for the given relationship names.

    When non-legacy edges are deleted from graph/vector DBs, their corresponding
    ledger entries (if any) should be marked as deleted so they are excluded from
    future legacy checks.
    """
    if not edge_relationship_names:
        return

    stmt = (
        update(GraphRelationshipLedger)
        .where(
            and_(
                GraphRelationshipLedger.deleted_at.is_(None),
                GraphRelationshipLedger.node_label.is_(None),
                or_(
                    *[
                        GraphRelationshipLedger.creator_function.ilike(f"%{name}")
                        for name in edge_relationship_names
                    ]
                ),
            )
        )
        .values(deleted_at=datetime.now(timezone.utc))
    )
    await session.execute(stmt)
    await session.commit()
