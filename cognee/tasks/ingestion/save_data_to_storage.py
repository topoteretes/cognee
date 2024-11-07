from typing import Union, BinaryIO
from cognee.modules.ingestion import save_data_to_file
from .transform_data import from_llama_index_format
from llama_index.core import Document

def save_data_to_storage(data: Union[BinaryIO, Document, str], dataset_name) -> list[str]:
    if not isinstance(data, list):
        # Convert data to a list as we work with lists further down.
        data = [data]

    file_paths = []

    for data_item in data:
        # data is a file object coming from upload.
        if hasattr(data_item, "file"):
            file_path = save_data_to_file(data_item.file, dataset_name, filename = data_item.filename)
            file_paths.append(file_path)

        # Check if data is of type Document or any of it's subclasses
        if isinstance(data_item, Document):
            file_paths.append(from_llama_index_format(data_item))

        if isinstance(data_item, str):
            # data is a file path
            if data_item.startswith("file://") or data_item.startswith("/"):
                file_paths.append(data_item.replace("file://", ""))

            # data is text
            else:
                file_path = save_data_to_file(data_item, dataset_name)
                file_paths.append(file_path)

    return file_paths
