import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname
from typing import Union, BinaryIO, Any

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion import save_data_to_file
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path
from cognee.shared.logging_utils import get_logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.tasks.web_scraper.utils import fetch_page_content
from cognee.tasks.web_scraper.ssrf_protection import validate_outbound_url
from cognee.tasks.ingestion.data_item import DataItem


logger = get_logger()


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = SaveDataSettings()


def _resolve_local_file_uri(data_item: str | Path, *, must_exist: bool = True) -> str | None:
    try:
        local_path = resolve_local_path(data_item, must_exist=must_exist)
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Data item could not be evaluated as a local file path.")
        return None
    except ValueError as error:
        raise IngestionError(message="Local file path is outside allowed roots.") from error

    return local_path.as_uri() if local_path.is_file() else None


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

        # data is s3 file path
        if parsed_url.scheme == "s3":
            return data_item
        elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
            # Guard against SSRF: reject disabled outbound HTTP, non-http(s) schemes,
            # and hosts that resolve to internal/reserved addresses before fetching.
            await validate_outbound_url(data_item)
            urls_to_page_contents = await fetch_page_content(data_item)
            return await save_data_to_file(urls_to_page_contents[data_item], file_extension="html")
        # data is local file path
        elif parsed_url.scheme == "file":
            if settings.accept_local_file_path:
                local_uri = _resolve_local_file_uri(url2pathname(parsed_url.path))
                if local_uri:
                    return local_uri
                raise IngestionError(message="Local file does not exist or is not a file.")
            else:
                raise IngestionError(message="Local files are not accepted.")

        # data is an absolute file path
        elif data_item.startswith("/") or (
            os.name == "nt" and len(data_item) > 1 and data_item[1] == ":"
        ):
            # Handle both Unix absolute paths (/path) and Windows absolute paths (C:\path)
            if settings.accept_local_file_path:
                local_uri = _resolve_local_file_uri(data_item)
                if local_uri:
                    return local_uri
                raise IngestionError(message="Local file does not exist or is not a file.")
            else:
                raise IngestionError(message="Local files are not accepted.")
        # Data is a relative file path
        local_uri = _resolve_local_file_uri(data_item)
        if local_uri:
            if settings.accept_local_file_path:
                return local_uri
            raise IngestionError(message="Local files are not accepted.")

        # data is text, save it to data storage and return the file path
        return await save_data_to_file(data_item)

    if isinstance(data_item, DataItem):
        # If instance is DataItem use the underlying data
        return await save_data_item_to_storage(data_item.data)

    # data is not a supported type
    raise IngestionError(message=f"Data type not supported: {type(data_item)}")
