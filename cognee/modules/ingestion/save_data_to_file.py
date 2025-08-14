from typing import BinaryIO, Union
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from .classify import classify


async def save_data_to_file(data: Union[str, BinaryIO], filename: str = None):
    storage_config = get_storage_config()

    data_root_directory = storage_config["data_root_directory"]
    classified_data = classify(data, filename)
    file_metadata = classified_data.get_metadata()

    storage = get_file_storage(data_root_directory)

    full_file_path = await storage.store("text_" + file_metadata["content_hash"] + ".txt", data)

    return full_file_path
