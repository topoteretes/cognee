import os
from urllib.parse import urlparse
from typing import Any, List, Tuple
from pathlib import Path
import tempfile

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


async def pull_from_s3(file_path, destination_file) -> None:
    async with open_data_file(file_path) as file:
        while True:
            chunk = file.read(8192)
            if not chunk:
                break
            destination_file.write(chunk)


async def data_item_to_text_file(
    data_item_path: str,
    preferred_loaders: dict[str, dict[str, Any]] = None,
) -> Tuple[str, LoaderInterface]:
    if isinstance(data_item_path, str):
        parsed_url = urlparse(data_item_path)

        # data is s3 file path
        if parsed_url.scheme == "s3":
            # TODO: Rework this to work with file streams and not saving data to temp storage
            # Note: proper suffix information is needed for OpenAI to handle mp3 files
            path_info = Path(parsed_url.path)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=path_info.suffix) as temp_file:
                await pull_from_s3(data_item_path, temp_file)
                temp_file.flush()  # Data needs to be saved to local storage
                loader = get_loader_engine()
                return await loader.load_file(temp_file.name, preferred_loaders), loader.get_loader(
                    temp_file.name, preferred_loaders
                )

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
