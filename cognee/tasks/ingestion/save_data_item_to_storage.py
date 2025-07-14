import os
from typing import Union, BinaryIO, Any

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file
from pydantic_settings import BaseSettings, SettingsConfigDict


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = True

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
        elif data_item.startswith("/") or (
            os.name == "nt" and len(data_item) > 1 and data_item[1] == ":"
        ):
            # Handle both Unix absolute paths (/path) and Windows absolute paths (C:\path)
            if settings.accept_local_file_path:
                # Normalize path separators before creating file URL
                normalized_path = os.path.normpath(data_item)
                # Use forward slashes in file URLs for consistency
                url_path = normalized_path.replace(os.sep, "/")
                file_path = "file://" + url_path
            else:
                raise IngestionError(message="Local files are not accepted.")
        # data is text
        else:
            file_path = await save_data_to_file(data_item)
    else:
        raise IngestionError(message=f"Data type not supported: {type(data_item)}")

    return file_path
