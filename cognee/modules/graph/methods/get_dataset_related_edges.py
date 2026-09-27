from uuid import UUID

from sqlalchemy import and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def get_dataset_related_edges(dataset_id: UUID, session: AsyncSession):
    return (await session.scalars(select(Edge).where(Edge.dataset_id == dataset_id))).all()


@with_async_session
async def get_global_dataset_related_edges(dataset_id: UUID, session: AsyncSession):
    EdgeAlias = aliased(Edge)

    subq = select(EdgeAlias.id).where(
        and_(
            EdgeAlias.slug == Edge.slug,
            EdgeAlias.dataset_id != dataset_id,
        )
    )

    query_statement = select(Edge).where(
        and_(
            Edge.dataset_id == dataset_id,
            ~exists(subq),
        )
    )

    related_edges = await session.scalars(query_statement)
    return related_edges.all()
