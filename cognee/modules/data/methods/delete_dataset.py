from cognee.modules.users.models import DatasetDatabase
from sqlalchemy import select

from cognee.modules.data.models import Dataset
from cognee.infrastructure.databases.utils.get_vector_dataset_database_handler import (
    get_vector_dataset_database_handler,
)
from cognee.infrastructure.databases.utils.get_graph_dataset_database_handler import (
    get_graph_dataset_database_handler,
)
from cognee.infrastructure.databases.relational import get_relational_engine


async def delete_dataset(dataset: Dataset):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        stmt = select(DatasetDatabase).where(
            DatasetDatabase.dataset_id == dataset.id,
        )
        dataset_database: DatasetDatabase = await session.scalar(stmt)
        if dataset_database:
            graph_dataset_database_handler = get_graph_dataset_database_handler(dataset_database)
            vector_dataset_database_handler = get_vector_dataset_database_handler(dataset_database)
            await graph_dataset_database_handler["handler_instance"].delete_dataset(
                dataset_database
            )
            await vector_dataset_database_handler["handler_instance"].delete_dataset(
                dataset_database
            )
    # TODO: Remove dataset from pipeline_run_status in Data objects related to dataset as well
    #       This blocks recreation of the dataset with the same name and data after deletion as
    #       it's marked as completed and will be just skipped even though it's empty.
    return await db_engine.delete_entity_by_id(dataset.__tablename__, dataset.id)
