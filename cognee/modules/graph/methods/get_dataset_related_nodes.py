from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_dataset_related_nodes(dataset_id: UUID, session: AsyncSession):
    query_statement = select(Node).where(Node.dataset_id == dataset_id)

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()
