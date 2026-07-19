from uuid import UUID
from sqlalchemy import select
from cognee.modules.graph.models import Node
from cognee.infrastructure.databases.relational import with_async_session


@with_async_session
async def get_dataset_node_ids(dataset_ids: list[UUID], session) -> set[str]:
    if not dataset_ids:
        return set()
    stmt = select(Node.slug).where(Node.dataset_id.in_(dataset_ids))
    result = await session.execute(stmt)
    return {str(slug) for slug in result.scalars().all()}
