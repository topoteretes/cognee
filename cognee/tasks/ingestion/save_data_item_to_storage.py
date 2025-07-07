from typing import Union, BinaryIO, Any
import os
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file


async def save_data_item_to_storage(data_item: Union[BinaryIO, str, Any], dataset_name: str) -> str:
    if "llama_index" in str(type(data_item)):
        # Dynamic import is used because the llama_index module is optional.
        from .transform_data import get_data_from_llama_index

        file_path = get_data_from_llama_index(data_item, dataset_name)

    # data is a file object coming from upload.
    elif hasattr(data_item, "file"):
        file_path = save_data_to_file(data_item.file, filename=data_item.filename)

    # elif isinstance(data_item, str):
    #     if data_item.startswith("s3://"):
    #         file_path = data_item
    #     # data is a file path
    #     elif data_item.startswith("file://") or data_item.startswith("/"):
    #         file_path = data_item.replace("file://", "")
    #     # data is text
    #     else:
    #         file_path = save_data_to_file(data_item)
    elif isinstance(data_item, str):
        # ---------- NEW, SAFER ROUTING ----------
        if data_item.startswith("s3://"):
            file_path = data_item

        elif data_item.startswith("file://"):
            local = data_item.replace("file://", "")
            # chỉ coi là file nếu thật sự tồn tại
            file_path = local if os.path.isfile(local) else save_data_to_file(data_item)

        elif data_item.startswith("/"):
            file_path = data_item if os.path.isfile(data_item) else save_data_to_file(data_item)

        else:
            file_path = save_data_to_file(data_item)
    else:
        raise IngestionError(message=f"Data type not supported: {type(data_item)}")

    return file_path
