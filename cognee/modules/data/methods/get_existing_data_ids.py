from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.data.models.Data import Data

# One IN carries at most this many ids. SQLite refuses a statement with more than 32,766 bind
# variables and asyncpg one with more than 32,767; 500 stays far below both.
DATA_ID_BATCH_SIZE = 500


async def get_existing_data_ids(
    session: AsyncSession, data_ids: list[UUID], dataset_id: UUID | None = None
) -> set[UUID]:
    """The ids in ``data_ids`` that have a ``Data`` row; with ``dataset_id``, only rows of it."""
    existing: set[UUID] = set()
    for start in range(0, len(data_ids), DATA_ID_BATCH_SIZE):
        query = select(Data.id).filter(Data.id.in_(data_ids[start : start + DATA_ID_BATCH_SIZE]))
        if dataset_id is not None:
            query = query.filter(Data.dataset_id == dataset_id)
        existing.update((await session.execute(query)).scalars())
    return existing
