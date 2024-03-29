import logging
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.data import Dataset, Data
from cognee.infrastructure.files import remove_file_from_storage
from cognee.infrastructure.databases.relational import DatabaseEngine

logger = logging.getLogger(__name__)

async def add_data_to_dataset(dataset: Dataset, data: Data):
    db_engine: DatabaseEngine = infrastructure_config.get_config()["database_engine"]

    existing_dataset = (await db_engine.query_entity(dataset)).scalar()
    existing_data = (await db_engine.query_entity(data)).scalar()

    if existing_dataset:
        if existing_data:
            await remove_old_raw_data(existing_data.raw_data_location)

            def update_raw_data():
                existing_data.raw_data_location = data.raw_data_location

            await db_engine.update(update_raw_data)

            if existing_dataset.id == dataset.id and dataset.name is not None:
                def update_name(): existing_dataset.name = dataset.name
                await db_engine.update(update_name)
        else:
            await db_engine.update(lambda: existing_dataset.data.append(data))
    else:
        if existing_data:
            await remove_old_raw_data(existing_data.raw_data_location)

            existing_data.raw_data_location = data.raw_data_location

            await db_engine.update(lambda: dataset.data.append(existing_data))
        else:
            await db_engine.update(lambda: dataset.data.append(data))

        await db_engine.create(dataset)

async def remove_old_raw_data(data_location: str):
    try:
        await remove_file_from_storage(data_location)
    except Exception:
        logger.error("Failed to remove old raw data file: %s", data_location)
