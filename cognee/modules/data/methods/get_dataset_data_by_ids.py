from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods.get_existing_data_ids import id_batches
from cognee.modules.data.models import Data


async def get_dataset_data_by_ids(dataset_id: UUID, data_ids: list[UUID]) -> list[Data]:
    """The dataset's Data rows among ``data_ids``, a batch of ids per query.

    Queried on the primary key alone and checked for the dataset here: with ``dataset_id``
    in the query, SQLite reads every row of the dataset through its index per batch.
    """
    rows: list[Data] = []
    async with get_relational_engine().get_async_session() as session:
        for batch in id_batches(data_ids):
            for row in await session.scalars(select(Data).where(Data.id.in_(batch))):
                if row.dataset_id == dataset_id:
                    rows.append(row)
    return rows
