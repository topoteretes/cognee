from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def get_dataset_related_edges(dataset_id: UUID, session: AsyncSession):
    return (
        await session.scalars(
            select(Edge).where(Edge.dataset_id == dataset_id).distinct(Edge.relationship_name)
        )
    ).all()
