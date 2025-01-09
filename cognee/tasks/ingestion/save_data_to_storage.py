from typing import Union, BinaryIO
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage


def save_data_to_storage(data: Union[BinaryIO, str], dataset_name) -> list[str]:
    if not isinstance(data, list):
        # Convert data to a list as we work with lists further down.
        data = [data]

    file_paths = []

    for data_item in data:
        file_path = save_data_item_to_storage(data_item, dataset_name)
        file_paths.append(file_path)

    return file_paths
