from typing import Union, BinaryIO, Any

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file
from pydantic_settings import BaseSettings, SettingsConfigDict


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = SaveDataSettings()


async def save_data_item_to_storage(data_item: Union[BinaryIO, str, Any]) -> str:
    if "llama_index" in str(type(data_item)):
        # Dynamic import is used because the llama_index module is optional.
        from .transform_data import get_data_from_llama_index

        file_path = await get_data_from_llama_index(data_item)

    # data is a file object coming from upload.
    elif hasattr(data_item, "file"):
        file_path = await save_data_to_file(data_item.file, filename=data_item.filename)

    elif isinstance(data_item, str):
        # data is s3 file or local file path
        if data_item.startswith("s3://") or data_item.startswith("file://"):
            file_path = data_item
        # data is a file path
        elif data_item.startswith("/") and settings.accept_local_file_path == "true":
            # TODO: Add check if ACCEPT_LOCAL_FILE_PATH is enabled, if it's not raise an error
            file_path = "file://" + data_item
        # data is text
        else:
            file_path = await save_data_to_file(data_item)
    else:
        raise IngestionError(message=f"Data type not supported: {type(data_item)}")

    return file_path
