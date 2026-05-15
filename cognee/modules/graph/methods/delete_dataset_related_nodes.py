from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node

BATCH_SIZE = 1000


@with_async_session
async def delete_dataset_related_nodes(dataset_id: UUID, session: AsyncSession):
    nodes = (await session.scalars(select(Node).where(Node.dataset_id == dataset_id))).all()

    node_ids = [node.id for node in nodes]
    for start_index in range(0, len(node_ids), BATCH_SIZE):
        node_id_batch = node_ids[start_index : start_index + BATCH_SIZE]
        await session.execute(delete(Node).where(Node.id.in_(node_id_batch)))
