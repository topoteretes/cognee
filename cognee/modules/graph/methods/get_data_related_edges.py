from uuid import UUID
from typing import Optional

from sqlalchemy.orm import aliased
from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge


@with_async_session
async def get_data_related_edges(dataset_id: UUID, data_id: UUID, session: AsyncSession):
    """Return edges owned by (dataset_id, data_id) that no other data in
    the same dataset owns. Used in multi-user mode, same semantics as
    `get_data_related_nodes`."""
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
async def get_global_data_related_edges(
    data_id: UUID,
    session: AsyncSession,
    dataset_id: Optional[UUID] = None,
):
    """Return edges safe to hard-delete for a single-DB (non-multi-user)
    deployment where the same data item may be linked to multiple datasets.
    Mirrors `get_global_data_related_nodes` — see that function's docstring
    for the dataset-aware shared-ownership semantics."""
    EdgeAlias = aliased(Edge)

    if dataset_id is None:
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
    else:
        subq = select(EdgeAlias.id).where(
            and_(
                EdgeAlias.slug == Edge.slug,
                or_(EdgeAlias.dataset_id != dataset_id, EdgeAlias.data_id != data_id),
            )
        )
        query_statement = select(Edge).where(
            and_(
                Edge.dataset_id == dataset_id,
                Edge.data_id == data_id,
                ~exists(subq),
            )
        )

    data_related_edges = await session.scalars(query_statement)
    return data_related_edges.all()
