import string
import random
from typing import BinaryIO, Union
from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import LocalStorage
from .classify import classify

def save_data_to_file(data: Union[str, BinaryIO], dataset_name: str, filename: str = None):
    base_config = get_base_config()
    data_directory_path = base_config.data_root_directory

    classified_data = classify(data, filename)

    storage_path = data_directory_path + "/" + dataset_name.replace(".", "/")
    LocalStorage.ensure_directory_exists(storage_path)

    file_metadata = classified_data.get_metadata()
    if "name" not in file_metadata or file_metadata["name"] is None:
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(32))
        file_metadata["name"] = "text_" + random_string + ".txt"
    file_name = file_metadata["name"]
    LocalStorage(storage_path).store(file_name, classified_data.get_data())

    return "file://" + storage_path + "/" + file_name
