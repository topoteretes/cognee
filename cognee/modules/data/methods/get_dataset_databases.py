from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.migrations.constants import GLOBAL_DATASET_ID
from cognee.modules.users.models import DatasetDatabase


async def get_dataset_databases() -> list[DatasetDatabase]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset_databases = (
            await session.scalars(
                # Exclude the reserved global sentinel row (used when access
                # control is off). It is not a real per-dataset database, so the
                # per-dataset migration/iteration paths must not process it
                # (e.g. after toggling ENABLE_BACKEND_ACCESS_CONTROL off->on).
                select(DatasetDatabase).where(DatasetDatabase.dataset_id != GLOBAL_DATASET_ID)
            )
        ).all()
        return dataset_databases
