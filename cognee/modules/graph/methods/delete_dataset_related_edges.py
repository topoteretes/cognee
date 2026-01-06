from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def delete_dataset_related_edges(dataset_id: UUID, session: AsyncSession):
    edges = (await session.scalars(select(Edge).where(Edge.dataset_id == dataset_id))).all()

    await session.execute(delete(Edge).where(Edge.id.in_([edge.id for edge in edges])))
