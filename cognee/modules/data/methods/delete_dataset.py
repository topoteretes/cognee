from cognee.modules.data.models import Dataset
from cognee.infrastructure.databases.relational import get_relational_engine


async def delete_dataset(dataset: Dataset):
    db_engine = get_relational_engine()

    return await db_engine.delete_entity_by_id(dataset.__tablename__, dataset.id)
