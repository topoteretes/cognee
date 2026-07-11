from uuid import UUID
from typing import Optional

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import DatasetDatabase


async def get_dataset_database(dataset_id: UUID) -> Optional[DatasetDatabase]:
    """Return the dataset's own-database mapping, or None when the dataset
    lives in the shared databases."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        return await session.scalar(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
        )
