import os
from urllib.parse import urlparse
from typing import Any, Tuple

from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.infrastructure.loaders import get_loader_engine
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.utils.open_data_file import open_data_file

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = get_logger(__name__)


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = SaveDataSettings()


async def data_item_to_text_file(
    data_item_path: str,
    preferred_loaders: dict[str, dict[str, Any]] = None,
) -> Tuple[str, LoaderInterface]:
    if isinstance(data_item_path, str):
        parsed_url = urlparse(data_item_path)

        # data is s3 file path
        if parsed_url.scheme == "s3":
            loader = get_loader_engine()
            async with open_data_file(data_item_path, mode="rb") as file:
                return await loader.load_file_stream(file, data_item_path, preferred_loaders)

        # data is local file path
        elif parsed_url.scheme == "file":
            if settings.accept_local_file_path:
                loader = get_loader_engine()
                return await loader.load_file(data_item_path, preferred_loaders), loader.get_loader(
                    data_item_path, preferred_loaders
                )
            else:
                raise IngestionError(message="Local files are not accepted.")

        # data is an absolute file path
        elif data_item_path.startswith("/") or (
            os.name == "nt" and len(data_item_path) > 1 and data_item_path[1] == ":"
        ):
            # Handle both Unix absolute paths (/path) and Windows absolute paths (C:\path)
            if settings.accept_local_file_path:
                loader = get_loader_engine()
                return await loader.load_file(data_item_path, preferred_loaders), loader.get_loader(
                    data_item_path, preferred_loaders
                )
            else:
                raise IngestionError(message="Local files are not accepted.")
    # data is not a supported type
    raise IngestionError(message=f"Data type not supported: {type(data_item_path)}")
