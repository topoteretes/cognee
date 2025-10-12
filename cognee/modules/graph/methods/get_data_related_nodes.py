from uuid import UUID
from sqlalchemy import and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_data_related_nodes(dataset_id: UUID, data_id: UUID, session: AsyncSession):
    NodeAlias = Node.__table__.alias("n2")

    subq = select(NodeAlias.c.id).where(
        and_(
            NodeAlias.c.slug == Node.slug,
            NodeAlias.c.dataset_id == dataset_id,
            NodeAlias.c.data_id != data_id,
        )
    )

    query_statement = select(Node).where(
        and_(Node.data_id == data_id, Node.dataset_id == dataset_id, ~exists(subq))
    )

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()
