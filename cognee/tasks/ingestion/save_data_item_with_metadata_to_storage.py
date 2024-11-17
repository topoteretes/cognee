from typing import Union, BinaryIO, Any
from cognee.modules.ingestion import save_data_to_file

def save_data_item_with_metadata_to_storage(data_item: Union[BinaryIO, str, Any], dataset_name: str) -> str:
    # Dynamic import is used because the llama_index module is optional. 
    # For the same reason Any is accepted as a data item
    from llama_index.core import Document
    from .transform_data import get_data_from_llama_index

    # Check if data is of type Document or any of it's subclasses
    if isinstance(data_item, Document):
        file_path = get_data_from_llama_index(data_item, dataset_name)

    # data is a file object coming from upload.
    elif hasattr(data_item, "file"):
        file_path = save_data_to_file(data_item.file, dataset_name, filename=data_item.filename)

    elif isinstance(data_item, str):
        # data is a file path
        if data_item.startswith("file://") or data_item.startswith("/"):
            file_path = data_item.replace("file://", "")
        # data is text
        else:
            file_path = save_data_to_file(data_item, dataset_name)
    else:
        raise ValueError(f"Data type not supported: {type(data_item)}")

    return file_path