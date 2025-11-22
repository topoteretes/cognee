from typing import List
from uuid import UUID
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node
from .GraphRelationshipLedger import GraphRelationshipLedger


@with_async_session
async def has_nodes_in_legacy_ledger(nodes: List[Node], user_id: UUID, session: AsyncSession):
    node_ids = [node.slug for node in nodes]

    query = select(
        GraphRelationshipLedger.source_node_id,
        GraphRelationshipLedger.destination_node_id,
    ).where(
        and_(
            GraphRelationshipLedger.user_id == user_id,
            or_(
                GraphRelationshipLedger.source_node_id.in_(node_ids),
                GraphRelationshipLedger.destination_node_id.in_(node_ids),
            ),
        )
    )

    legacy_nodes = await session.execute(query)
    entries = legacy_nodes.all()

    found_ids = set()
    for entry in entries:
        found_ids.add(entry.source_node_id)
        found_ids.add(entry.destination_node_id)

    return [node_id in found_ids for node_id in node_ids]
