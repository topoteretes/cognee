from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge

BATCH_SIZE = 1000


@with_async_session
async def delete_dataset_related_edges(dataset_id: UUID, session: AsyncSession):
    nodes = (await session.scalars(select(Edge).where(Edge.dataset_id == dataset_id))).all()

    edge_ids = [node.id for node in nodes]
    for start_index in range(0, len(edge_ids), BATCH_SIZE):
        edge_id_batch = edge_ids[start_index : start_index + BATCH_SIZE]
        await session.execute(delete(Edge).where(Edge.id.in_(edge_id_batch)))
