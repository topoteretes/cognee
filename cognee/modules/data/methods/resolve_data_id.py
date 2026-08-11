from typing import Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data


async def resolve_data_id(dataset_id: UUID, data_id: UUID) -> Optional[UUID]:
    """Resolve a caller-supplied data_id to the canonical row id in a dataset.

    Users hold data_ids in external mappings, so every id ever issued keeps
    resolving: the exact id wins (the normal case — ids are preserved through
    updates and by the backfill); on a miss, the id is matched against
    ``legacy_id`` — the recorded original of a row whose identity
    forked when a pre-refactor shared row was split per dataset. Within one
    dataset that fork target is unique, so the fallback is deterministic.

    Returns the canonical id, or None when the id is unknown in this dataset.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        exact = (
            await session.execute(
                select(Data.id).filter(Data.id == data_id, Data.dataset_id == dataset_id)
            )
        ).scalar_one_or_none()
        if exact:
            return exact

        return (
            await session.execute(
                select(Data.id).filter(Data.legacy_id == data_id, Data.dataset_id == dataset_id)
            )
        ).scalar_one_or_none()
