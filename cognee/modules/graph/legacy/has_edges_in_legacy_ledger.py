from typing import List
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge
from .GraphRelationshipLedger import GraphRelationshipLedger


@with_async_session
async def has_edges_in_legacy_ledger(edges: List[Edge], session: AsyncSession):
    if len(edges) == 0:
        return []

    query = select(GraphRelationshipLedger).where(
        and_(
            GraphRelationshipLedger.node_label.is_(None),
            or_(
                *[
                    GraphRelationshipLedger.creator_function.ilike(f"%{edge.relationship_name}")
                    for edge in edges
                ]
            ),
        )
    )

    legacy_edges = (await session.scalars(query)).all()

    legacy_edge_names = set([edge.creator_function.split(".")[1] for edge in legacy_edges])

    return [edge.relationship_name in legacy_edge_names for edge in edges]
