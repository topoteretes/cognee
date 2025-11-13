from uuid import UUID
from sqlalchemy.orm import aliased
from sqlalchemy import and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_data_related_nodes(dataset_id: UUID, data_id: UUID, session: AsyncSession):
    NodeAlias = aliased(Node)

    subq = select(NodeAlias.id).where(
        and_(
            NodeAlias.slug == Node.slug,
            NodeAlias.dataset_id == dataset_id,
            NodeAlias.data_id != data_id,
        )
    )

    query_statement = select(Node).where(
        and_(Node.data_id == data_id, Node.dataset_id == dataset_id, ~exists(subq))
    )

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()


@with_async_session
async def get_global_data_related_nodes(data_id: UUID, session: AsyncSession):
    NodeAlias = aliased(Node)

    subq = select(NodeAlias.id).where(
        and_(
            NodeAlias.slug == Node.slug,
            NodeAlias.data_id != data_id,
        )
    )

    query_statement = select(Node).where(and_(Node.data_id == data_id, ~exists(subq)))

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()
