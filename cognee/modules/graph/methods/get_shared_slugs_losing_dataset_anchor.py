from typing import List
from uuid import UUID

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_shared_slugs_losing_dataset_anchor(
    dataset_id: UUID,
    data_id: UUID,
    session: AsyncSession,
) -> List[UUID]:
    """Return slugs that will lose their link to `dataset_id` once
    `(dataset_id, data_id)` is removed from the ledger, but will still
    exist in the graph because another `(dataset_id_other, data_id_other)`
    pair owns them.

    These are the shared slugs whose stored `belongs_to_set` array still
    carries the soon-to-be-removed dataset's name even though no surviving
    ledger row anchors them to it anymore — callers reconcile the array
    with a targeted `remove_belongs_to_set_tags([dataset_name], node_ids=...)`
    pass so the property stays in sync with the edges.

    Intentionally excludes slugs that are uniquely owned by
    `(dataset_id, data_id)`: those are hard-deleted upstream by
    `delete_from_graph_and_vector` and don't need detagging.
    """
    SameDataset = aliased(Node)
    OtherOwner = aliased(Node)

    loses_dataset_anchor = ~exists(
        select(SameDataset.id).where(
            and_(
                SameDataset.slug == Node.slug,
                SameDataset.dataset_id == dataset_id,
                SameDataset.data_id != data_id,
            )
        )
    )

    has_other_owner = exists(
        select(OtherOwner.id).where(
            and_(
                OtherOwner.slug == Node.slug,
                or_(
                    OtherOwner.dataset_id != dataset_id,
                    OtherOwner.data_id != data_id,
                ),
            )
        )
    )

    statement = (
        select(Node.slug)
        .where(
            and_(
                Node.dataset_id == dataset_id,
                Node.data_id == data_id,
                loses_dataset_anchor,
                has_other_owner,
            )
        )
        .distinct()
    )

    result = await session.scalars(statement)
    return list(result.all())
