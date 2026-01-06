from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def delete_data_related_edges(data_id: UUID, session: AsyncSession):
    edges = (await session.scalars(select(Edge).where(Edge.data_id == data_id))).all()

    await session.execute(delete(Edge).where(Edge.id.in_([edge.id for edge in edges])))

