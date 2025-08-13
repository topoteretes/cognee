import os
from urllib.parse import urlparse
from typing import Union, BinaryIO, Any, List

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file
from cognee.infrastructure.loaders import get_loader_engine
from pydantic_settings import BaseSettings, SettingsConfigDict


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = SaveDataSettings()


async def data_item_to_text_file(data_item_path: str, preferred_loaders: List[str]) -> str:
    if isinstance(data_item_path, str):
        parsed_url = urlparse(data_item_path)

        # data is s3 file path
        if parsed_url.scheme == "s3":
            # TODO: Add loader ingestion support for S3 files
            return data_item_path

        # data is local file path
        elif parsed_url.scheme == "file":
            if settings.accept_local_file_path:
                loader = get_loader_engine()
                content = await loader.load_file(data_item_path)
                return await save_data_to_file(content)
            else:
                raise IngestionError(message="Local files are not accepted.")

        # data is an absolute file path
        elif data_item_path.startswith("/") or (
            os.name == "nt" and len(data_item_path) > 1 and data_item_path[1] == ":"
        ):
            # Handle both Unix absolute paths (/path) and Windows absolute paths (C:\path)
            if settings.accept_local_file_path:
                loader = get_loader_engine()
                content = await loader.load_file(data_item_path)
                return await save_data_to_file(content)
            else:
                raise IngestionError(message="Local files are not accepted.")

    # data is not a supported type
    raise IngestionError(message=f"Data type not supported: {type(data_item_path)}")
