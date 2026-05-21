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

    relationship_names = list({edge.relationship_name for edge in edges})
    legacy_edge_names = set()
    # SQLite has a max expression tree depth; keep OR batches small.
    batch_size = 250

    for start_index in range(0, len(relationship_names), batch_size):
        relationship_batch = relationship_names[start_index : start_index + batch_size]
        query = select(GraphRelationshipLedger.creator_function).where(
            and_(
                GraphRelationshipLedger.node_label.is_(None),
                or_(
                    *[
                        GraphRelationshipLedger.creator_function.ilike(f"%{relationship_name}")
                        for relationship_name in relationship_batch
                    ]
                ),
            )
        )
        creator_functions = (await session.scalars(query)).all()

        for creator_function in creator_functions:
            legacy_edge_names.add(creator_function.split(".")[1])

    return [edge.relationship_name in legacy_edge_names for edge in edges]
