from uuid import UUID
from sqlalchemy.orm import aliased
from sqlalchemy import and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def get_data_related_edges(dataset_id: UUID, data_id: UUID, session: AsyncSession):
    EdgeAlias = aliased(Edge)

    subq = select(EdgeAlias.id).where(
        and_(
            EdgeAlias.slug == Edge.slug,
            EdgeAlias.dataset_id == dataset_id,
            EdgeAlias.data_id != data_id,
        )
    )

    query_statement = select(Edge).where(
        and_(
            Edge.data_id == data_id,
            Edge.dataset_id == dataset_id,
            ~exists(subq),
        )
    )

    data_related_edges = await session.scalars(query_statement)
    return data_related_edges.all()


@with_async_session
async def get_global_data_related_edges(data_id: UUID, session: AsyncSession):
    EdgeAlias = aliased(Edge)

    subq = select(EdgeAlias.id).where(
        and_(
            EdgeAlias.slug == Edge.slug,
            EdgeAlias.data_id != data_id,
        )
    )

    query_statement = select(Edge).where(
        and_(
            Edge.data_id == data_id,
            ~exists(subq),
        )
    )

    data_related_edges = await session.scalars(query_statement)
    return data_related_edges.all()
