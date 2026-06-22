from uuid import UUID
from typing import Optional

from sqlalchemy.orm import aliased
from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_data_related_nodes(dataset_id: UUID, data_id: UUID, session: AsyncSession):
    """Return nodes owned by (dataset_id, data_id) that no other data in the
    same dataset owns. Used in multi-user mode where each dataset has its
    own graph/vector database, so cross-dataset slug collisions cannot
    occur and only intra-dataset sharing is considered."""
    NodeAlias = aliased(Node)

    subq = select(NodeAlias.id).where(
        and_(
            NodeAlias.slug == Node.slug,
            NodeAlias.dataset_id == dataset_id,
            NodeAlias.data_id != data_id,
        )
    )

    query_statement = select(Node).where(
        and_(Node.data_id == data_id, Node.dataset_id == dataset_id, ~exists(subq))
    )

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()


@with_async_session
async def get_global_data_related_nodes(
    data_id: UUID,
    session: AsyncSession,
    dataset_id: Optional[UUID] = None,
):
    """Return nodes safe to hard-delete for a single-DB (non-multi-user)
    deployment where the same data item may be linked to multiple datasets.

    When `dataset_id` is provided, a slug is only considered removable if
    NO other `(dataset_id, data_id)` pair owns it — protects shared data
    items so deleting one dataset's link doesn't wipe the graph/vector
    rows that another dataset still references.

    When `dataset_id` is None, falls back to the legacy global semantics
    (slug is exclusive to this `data_id` globally) for callers that don't
    yet plumb the dataset through.
    """
    NodeAlias = aliased(Node)

    if dataset_id is None:
        subq = select(NodeAlias.id).where(
            and_(
                NodeAlias.slug == Node.slug,
                NodeAlias.data_id != data_id,
            )
        )
        query_statement = select(Node).where(and_(Node.data_id == data_id, ~exists(subq)))
    else:
        subq = select(NodeAlias.id).where(
            and_(
                NodeAlias.slug == Node.slug,
                or_(NodeAlias.dataset_id != dataset_id, NodeAlias.data_id != data_id),
            )
        )
        query_statement = select(Node).where(
            and_(
                Node.dataset_id == dataset_id,
                Node.data_id == data_id,
                ~exists(subq),
            )
        )

    data_related_nodes = await session.scalars(query_statement)
    return data_related_nodes.all()
