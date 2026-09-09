from uuid import UUID

from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data


async def get_dataset_data(
    dataset_id: UUID,
    limit: int | None = None,
    offset: int = 0,
) -> list[Data]:
    """Return a dataset's data rows.

    `limit` defaults to None -- every row -- because most callers here are
    internal pipeline code that genuinely needs the whole set (cognify,
    sync, the estimator, permission fan-out). The HTTP route is the caller
    that must not do that, and it passes a bounded limit of its own.
    """
    db_engine = get_relational_engine()

    query = select(Data).filter(Data.dataset_id == dataset_id).order_by(Data.data_size.desc())

    if offset:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)

    async with db_engine.get_async_session() as session:
        result = await session.execute(query)

        data = list(result.scalars().all())

        return data


async def count_dataset_data(dataset_id: UUID) -> int:
    """Count a dataset's data rows without fetching them.

    Callers that only need "how many documents" -- dataset list badges,
    upload/ingest polling -- used to read len() off the full row set, which
    on a large dataset meant transferring tens of megabytes to produce one
    integer. This is served by the existing dataset_id index instead.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(func.count()).select_from(Data).filter(Data.dataset_id == dataset_id)
        )

        return result.scalar_one()
