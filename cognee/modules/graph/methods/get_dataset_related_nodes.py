from uuid import UUID
from sqlalchemy.orm import aliased
from sqlalchemy import exists, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_dataset_related_nodes(dataset_id: UUID, session: AsyncSession):
    query_statement = select(Node).where(Node.dataset_id == dataset_id)

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()


@with_async_session
async def get_global_dataset_related_nodes(dataset_id: UUID, session: AsyncSession):
    NodeAlias = aliased(Node)

    subq = select(NodeAlias.id).where(
        and_(
            NodeAlias.slug == Node.slug,
            NodeAlias.dataset_id != dataset_id,
        )
    )

    query_statement = select(Node).where(
        and_(
            Node.dataset_id == dataset_id,
            ~exists(subq),
        )
    )

    related_nodes = await session.scalars(query_statement)
    return related_nodes.all()
