from uuid import UUID
from sqlalchemy import and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def delete_data_related_edges(data_id: UUID, dataset_id: UUID, session: AsyncSession):
    await session.execute(
        delete(Edge).where(and_(Edge.data_id == data_id, Edge.dataset_id == dataset_id))
    )
