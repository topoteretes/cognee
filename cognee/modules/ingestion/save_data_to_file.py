import os.path
import hashlib
from typing import BinaryIO, Union
from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import LocalStorage
from .classify import classify


def save_data_to_file(data: Union[str, BinaryIO], filename: str = None):
    base_config = get_base_config()
    data_directory_path = base_config.data_root_directory

    classified_data = classify(data, filename)

    storage_path = os.path.join(data_directory_path, "data")
    LocalStorage.ensure_directory_exists(storage_path)

    file_metadata = classified_data.get_metadata()
    if "name" not in file_metadata or file_metadata["name"] is None:
        data_contents = classified_data.get_data().encode("utf-8")
        hash_contents = hashlib.md5(data_contents).hexdigest()
        file_metadata["name"] = "text_" + hash_contents + ".txt"
    file_name = file_metadata["name"]

    # Don't save file if it already exists
    if not os.path.isfile(os.path.join(storage_path, file_name)):
        LocalStorage(storage_path).store(file_name, classified_data.get_data())

    return "file://" + storage_path + "/" + file_name
