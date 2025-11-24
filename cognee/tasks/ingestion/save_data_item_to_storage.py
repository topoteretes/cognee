import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Union, BinaryIO, Any

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file
from cognee.shared.logging_utils import get_logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.tasks.web_scraper.utils import fetch_page_content


logger = get_logger()


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = SaveDataSettings()


async def save_data_item_to_storage(data_item: Union[BinaryIO, str, Any]) -> str:
    if "llama_index" in str(type(data_item)):
        # Dynamic import is used because the llama_index module is optional.
        from .transform_data import get_data_from_llama_index

        return await get_data_from_llama_index(data_item)

    if "docling" in str(type(data_item)):
        from docling_core.types import DoclingDocument

        if isinstance(data_item, DoclingDocument):
            data_item = data_item.export_to_text()

    # data is a file object coming from upload.
    if hasattr(data_item, "file"):
        return await save_data_to_file(data_item.file, filename=data_item.filename)

    if isinstance(data_item, str):
        parsed_url = urlparse(data_item)

        try:
            # In case data item is a string with a relative path transform data item to absolute path and check
            # if the file exists
            abs_path = (Path.cwd() / Path(data_item)).resolve()
            abs_path.is_file()
        except (OSError, ValueError):
            # In case file path is too long it's most likely not a relative path
            abs_path = data_item
            logger.debug(f"Data item was too long to be a possible file path: {abs_path}")
            abs_path = Path("")

        # data is s3 file path
        if parsed_url.scheme == "s3":
            return data_item
        elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
            urls_to_page_contents = await fetch_page_content(data_item)
            return await save_data_to_file(urls_to_page_contents[data_item], file_extension="html")
        # data is local file path
        elif parsed_url.scheme == "file":
            if settings.accept_local_file_path:
                return data_item
            else:
                raise IngestionError(message="Local files are not accepted.")

        # data is an absolute file path
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

                return file_path
            else:
                raise IngestionError(message="Local files are not accepted.")
        # Data is a relative file path
        elif abs_path.is_file():
            if settings.accept_local_file_path:
                # Normalize path separators before creating file URL
                normalized_path = os.path.normpath(abs_path)
                # Use forward slashes in file URLs for consistency
                url_path = normalized_path.replace(os.sep, "/")
                file_path = "file://" + url_path
                return file_path

        # data is text, save it to data storage and return the file path
        return await save_data_to_file(data_item)

    # data is not a supported type
    raise IngestionError(message=f"Data type not supported: {type(data_item)}")
