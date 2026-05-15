from typing import List
from uuid import UUID

from sqlalchemy import and_, exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Node


@with_async_session
async def get_orphaned_nodeset_labels_for_dataset(
    dataset_id: UUID,
    data_id: UUID,
    session: AsyncSession,
) -> List[str]:
    """Return labels of NodeSet ledger rows owned by `(dataset_id, data_id)`
    that will have NO other `(dataset_id, *)` anchor after that row goes away.

    These are the NodeSet tag names the dataset is fully losing for this slice
    of the delete. Used as the tag list for the scoped detag on surviving
    shared slugs: any `belongs_to_set` entry on a co-owned slug that traces
    back to one of these NodeSets is stale w.r.t. this dataset and must be
    stripped from the graph/vector property.

    NodeSets that are uniquely owned globally by `(dataset_id, data_id)` are
    hard-deleted upstream by `delete_from_graph_and_vector`, which also does
    an unscoped detag for their labels. Including them here is harmless — the
    scoped pass becomes a no-op against an already-stripped property — so we
    do not exclude them.
    """
    SameDatasetOther = aliased(Node)

    has_other_same_dataset_anchor = exists(
        select(SameDatasetOther.id).where(
            and_(
                SameDatasetOther.slug == Node.slug,
                SameDatasetOther.dataset_id == dataset_id,
                SameDatasetOther.data_id != data_id,
            )
        )
    )

    statement = (
        select(Node.label)
        .where(
            and_(
                Node.dataset_id == dataset_id,
                Node.data_id == data_id,
                Node.type == "NodeSet",
                Node.label.is_not(None),
                ~has_other_same_dataset_anchor,
            )
        )
        .distinct()
    )

    result = await session.scalars(statement)
    return [label for label in result.all() if label]
