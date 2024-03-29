import asyncio
from uuid import UUID, uuid4
from cognee.infrastructure.data import Data, Dataset
from .add_data_to_dataset import add_data_to_dataset
from .data_types import IngestionData

async def save(dataset_id: UUID, dataset_name: str, data_id: UUID, data: IngestionData):
    file_path = uuid4().hex + "." + data.get_extension()

    promises = []

    promises.append(
        add_data_to_dataset(
            Dataset(
                id = dataset_id,
                name = dataset_name if dataset_name else dataset_id.hex
            ),
            Data(
                id = data_id,
                raw_data_location = file_path,
                name = data.metadata["name"],
                meta_data = data.metadata
            )
        )
    )

    await asyncio.gather(*promises)
