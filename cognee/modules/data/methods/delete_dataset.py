from cognee.modules.users.models import DatasetDatabase
from sqlalchemy import select

from cognee.modules.data.models import Dataset, DatasetData, Data
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

        # Clear pipeline_status entries for this dataset from related Data objects
        # so re-adding the same data isn't blocked by stale "completed" status.
        data_ids_query = select(DatasetData.data_id).where(DatasetData.dataset_id == dataset.id)
        data_records = (
            (await session.execute(select(Data).where(Data.id.in_(data_ids_query)))).scalars().all()
        )

        dataset_id_str = str(dataset.id)
        for data_record in data_records:
            if not data_record.pipeline_status:
                continue
            updated = False
            for pipeline_name in list(data_record.pipeline_status.keys()):
                if dataset_id_str in data_record.pipeline_status[pipeline_name]:
                    del data_record.pipeline_status[pipeline_name][dataset_id_str]
                    updated = True
            if updated:
                await session.merge(data_record)

        await session.commit()

    return await db_engine.delete_entity_by_id(dataset.__tablename__, dataset.id)
