from typing import Union, BinaryIO

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file


def save_data_item_to_storage(data_item: Union[BinaryIO, str], dataset_name: str) -> str:
    # data is a file object coming from upload.
    if hasattr(data_item, "file"):
        file_path = save_data_to_file(data_item.file, filename=data_item.filename)

    elif isinstance(data_item, str):
        # data is a file path
        if data_item.startswith("file://") or data_item.startswith("/"):
            file_path = data_item.replace("file://", "")
        # data is text
        else:
            file_path = save_data_to_file(data_item)
    else:
        raise IngestionError(message=f"Data type not supported: {type(data_item)}")

    return file_path
