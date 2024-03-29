import asyncio
from uuid import UUID, uuid4
from typing import Union, BinaryIO, List
import cognee.modules.ingestion as ingestion
from cognee.infrastructure import infrastructure_config

class DatasetException(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message


async def add_standalone(
    data: Union[str, BinaryIO, List[Union[str, BinaryIO]]],
    dataset_id: UUID = uuid4(),
    dataset_name: str = None
):
    db_engine = infrastructure_config.get_config()["database_engine"]
    if db_engine.is_db_done is not True:
        await db_engine.ensure_tables()

    if not data:
        raise DatasetException("Data must be provided to cognee.add(data: str)")

    if isinstance(data, list):
        promises = []

        for data_item in data:
            promises.append(add_standalone(data_item, dataset_id, dataset_name))

        results = await asyncio.gather(*promises)

        return results


    if is_data_path(data):
        with open(data.replace("file://", ""), "rb") as file:
            return await add_standalone(file, dataset_id, dataset_name)

    classified_data = ingestion.classify(data)

    data_id = ingestion.identify(classified_data)

    await ingestion.save(dataset_id, dataset_name, data_id, classified_data)

    return dataset_id

    # await ingestion.vectorize(dataset_id, dataset_name, data_id, classified_data)


def is_data_path(data: str) -> bool:
    return False if not isinstance(data, str) else data.startswith("file://")