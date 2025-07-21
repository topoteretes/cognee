import hashlib
from typing import BinaryIO, Union
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from .classify import classify


async def save_data_to_file(data: Union[str, BinaryIO], filename: str = None):
    storage_config = get_storage_config()

    data_root_directory = storage_config["data_root_directory"]

    classified_data = classify(data, filename)

    file_metadata = classified_data.get_metadata()

    async with classified_data.get_data() as data:
        if "name" not in file_metadata or file_metadata["name"] is None:
            data_contents = data.encode("utf-8")
            hash_contents = hashlib.md5(data_contents).hexdigest()
            file_metadata["name"] = "text_" + hash_contents + ".txt"

        file_name = file_metadata["name"]

        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(file_name, data)

        return full_file_path
