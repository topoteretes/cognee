from uuid import UUID
from typing import Dict, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from .GraphRelationshipLedger import GraphRelationshipLedger


@with_async_session
async def record_data_in_legacy_ledger(
    nodes: List[DataPoint],
    edges: List[Tuple[UUID, UUID, str, Dict]],
    session: AsyncSession,
) -> None:
    relationships = [
        GraphRelationshipLedger(
            source_node_id=node.id,
            destination_node_id=node.id,
            node_label=getattr(node, "name", getattr(node, "text", node.id)),
            creator_function="add_data_points.nodes",
        )
        for node in nodes
    ] + [
        GraphRelationshipLedger(
            source_node_id=edge[0],
            destination_node_id=edge[1],
            creator_function=f"add_data_points.{edge[2]}",
        )
        for edge in edges
    ]

    session.add_all(relationships)

    await session.commit()
