from uuid import UUID
from typing import Dict, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.users.models.User import User
from .GraphRelationshipLedger import GraphRelationshipLedger


@with_async_session
async def record_data_in_legacy_ledger(
    nodes: List[DataPoint],
    edges: List[Tuple[UUID, UUID, str, Dict]],
    user: User,
    session: AsyncSession,
) -> None:
    relationships = [
        GraphRelationshipLedger(
            source_node_id=node.id,
            destination_node_id=node.id,
            creator_function="add_nodes",
            user_id=user.id,
        )
        for node in nodes
    ] + [
        GraphRelationshipLedger(
            source_node_id=edge[0],
            destination_node_id=edge[1],
            creator_function=f"add_edges.{edge[2]}",
            user_id=user.id,
        )
        for edge in edges
    ]

    session.add_all(relationships)

    await session.commit()
