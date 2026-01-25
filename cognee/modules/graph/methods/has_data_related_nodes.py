from uuid import UUID
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def has_data_related_nodes(dataset_id: UUID, data_id: UUID, session: AsyncSession):
    query_statement = (
        select(Node).where(and_(Node.data_id == data_id, Node.dataset_id == dataset_id)).limit(1)
    )

    data_related_node = await session.scalar(query_statement)
    return data_related_node is not None
