from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def delete_dataset_related_nodes(dataset_id: UUID, session: AsyncSession):
    nodes = (await session.scalars(select(Node).where(Node.dataset_id == dataset_id))).all()

    await session.execute(delete(Node).where(Node.id.in_([node.id for node in nodes])))

