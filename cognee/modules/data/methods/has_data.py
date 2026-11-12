from uuid import UUID

from sqlalchemy import exists, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data


async def has_data(dataset_id: UUID, data_id: UUID) -> bool:
    """Whether a ``Data`` row with exactly this id exists in the dataset.

    A single-row existence query: no rows are loaded, so it is the right
    check after a delete (``datasets.delete_data`` verifies the row is gone
    with it), where listing the dataset would load every remaining row.
    Unlike ``resolve_data_id`` it does not fall back to ``legacy_id``.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = select(exists().where(Data.id == data_id, Data.dataset_id == dataset_id))
        return bool((await session.execute(query)).scalar())
