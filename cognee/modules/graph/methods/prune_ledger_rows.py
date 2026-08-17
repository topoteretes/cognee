"""Ledger cleanup for chunk-scoped deletions.

Lives beside ``delete_chunks_incremental`` because it completes the same
operation: that function removes nodes from the graph, this one removes the
rollback-ledger rows that referenced them. Keeping both in the module that owns
the ledger table means its delete semantics have one implementation, not a
second copy in a caller that its owners would not think to grep.
"""

from typing import List
from uuid import UUID

from sqlalchemy import and_, delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.graph.models import Node


async def prune_ledger_rows(data_id: UUID, dataset_id: UUID, doomed_ids: List[str]) -> None:
    """Drop rollback-ledger rows for nodes an incremental delete removed.

    Scoped to one document in one dataset: ledger rows are keyed by
    (data_id, dataset_id, slug), so a node id shared with another document's
    subgraph cannot be pruned out from under it.
    """
    if not doomed_ids:
        return

    slugs = [UUID(str(doomed)) for doomed in doomed_ids]
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        await session.execute(
            delete(Node).where(
                and_(Node.data_id == data_id, Node.dataset_id == dataset_id, Node.slug.in_(slugs))
            )
        )
        await session.commit()
