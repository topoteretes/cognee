from uuid import UUID
from sqlalchemy import and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def delete_data_related_nodes(data_id: UUID, dataset_id: UUID, session: AsyncSession):
    await session.execute(
        delete(Node).where(and_(Node.data_id == data_id, Node.dataset_id == dataset_id))
    )
