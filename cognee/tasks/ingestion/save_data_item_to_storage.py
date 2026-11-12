import os
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse
from urllib.request import url2pathname

from fastapi import status
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path
from cognee.modules.ingestion import StoredFile, save_data_to_file_detailed
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.operations.origin import ORIGIN_API, get_operation_origin
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.web_scraper.ssrf_protection import validate_outbound_url
from cognee.tasks.web_scraper.utils import fetch_page_content

logger = get_logger()


class SaveDataSettings(BaseSettings):
    # None = not set: the gate decides from the call origin. True or False, from
    # the environment or assigned in process, is explicit and wins everywhere.
    accept_local_file_path: bool | None = None

    model_config = SettingsConfigDict(extra="allow")


settings = SaveDataSettings()


def local_file_paths_configured() -> bool:
    """Whether ``ACCEPT_LOCAL_FILE_PATH`` was set, rather than left to the call origin."""
    return settings.accept_local_file_path is not None


def local_file_paths_accepted() -> bool:
    """Whether a path the caller passed may be read with this server's access.

    That is a local file or directory, including one under cognee's own data
    root, and an ``s3://`` object, read with cognee's AWS credentials. One
    rule, read at the gates: an explicit ``ACCEPT_LOCAL_FILE_PATH`` is
    honoured everywhere. Unset, the answer depends on who is calling: the SDK,
    CLI and MCP server run on the caller's own machine, so the code default
    (True) stands; an API request names a file on the server, so it is
    refused. Uploads and text are never subject to this: they are stored
    first and loaded from cognee's own copy.
    """
    if settings.accept_local_file_path is not None:
        return settings.accept_local_file_path
    return get_operation_origin() != ORIGIN_API


def require_server_path_access() -> None:
    """Raise unless the gate admits a path read with this server's own access.

    The message says which setting refused the path and how to change it; the
    status is 403, a policy refusal, not a problem with the data itself.
    """
    if local_file_paths_accepted():
        return
    if local_file_paths_configured():
        message = (
            "Server-side paths are not accepted (ACCEPT_LOCAL_FILE_PATH=false). Set "
            "ACCEPT_LOCAL_FILE_PATH=true in the Cognee environment (on the server for API "
            "calls) to allow them, or upload the file instead."
        )
    else:
        message = (
            "Server-side paths are not accepted for API requests by default because "
            "ACCEPT_LOCAL_FILE_PATH is unset. Set ACCEPT_LOCAL_FILE_PATH=true in the "
            "server's environment to allow them, or upload the file instead."
        )
    raise IngestionError(message=message, status_code=status.HTTP_403_FORBIDDEN)


def _resolve_local_file_uri(
    data_item: str | Path, *, must_exist: bool = True, strict: bool = False
) -> str | None:
    try:
        local_path = resolve_local_path(data_item, must_exist=must_exist)
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Data item could not be evaluated as a local file path.")
        return None
    except ValueError as error:
        if strict:
            raise IngestionError(message="Local file path is outside allowed roots.") from error
        # A path-looking string outside the allowed roots is never read as a file;
        # non-strict callers fall through to plain-text ingestion instead.
        return None

    return local_path.as_uri() if local_path.is_file() else None


async def save_data_item_to_storage_detailed(
    data_item: BinaryIO | str | Any,
) -> StoredFile:
    """Put ``data_item`` in cognee storage and describe what landed there.

    ``StoredFile.metadata`` is populated for every item whose bytes passed
    through this process (uploads, fetched pages, raw text) — it is computed
    while they are in hand, so ingestion never has to read the object back to
    hash it. Items that are already addressable storage (an ``s3://`` URL, a
    local path) are handed through untouched and carry no metadata.
    """
    if "llama_index" in str(type(data_item)):
        # Dynamic import is used because the llama_index module is optional.
        from .transform_data import get_data_from_llama_index

        return StoredFile(file_path=await get_data_from_llama_index(data_item))

    if "docling" in str(type(data_item)):
        from docling_core.types import DoclingDocument

        if isinstance(data_item, DoclingDocument):
            data_item = data_item.export_to_text()

    # data is a file object coming from upload.
    if hasattr(data_item, "file"):
        return await save_data_to_file_detailed(data_item.file, filename=data_item.filename)

    if isinstance(data_item, str):
        parsed_url = urlparse(data_item)

        # data is s3 file path
        if parsed_url.scheme == "s3":
            # Read with cognee's own AWS credentials, so it is a server-side
            # path like a local one: same gate.
            require_server_path_access()
            return StoredFile(file_path=data_item)
        elif parsed_url.scheme == "http" or parsed_url.scheme == "https":
            # Guard against SSRF: reject disabled outbound HTTP, non-http(s) schemes,
            # and hosts that resolve to internal/reserved addresses before fetching.
            await validate_outbound_url(data_item)
            urls_to_page_contents = await fetch_page_content(data_item)
            return await save_data_to_file_detailed(
                urls_to_page_contents[data_item], file_extension="html"
            )
        # data is local file path
        elif parsed_url.scheme == "file":
            require_server_path_access()
            local_uri = _resolve_local_file_uri(url2pathname(parsed_url.path), strict=True)
            if local_uri:
                return StoredFile(file_path=local_uri)
            raise IngestionError(message="Local file does not exist or is not a file.")

        # data is an absolute file path that points at an existing file
        elif (
            data_item.startswith("/")
            or (os.name == "nt" and len(data_item) > 1 and data_item[1] == ":")
        ) and Path(os.path.normpath(data_item)).is_absolute():
            # Handle both Unix absolute paths (/path) and Windows absolute paths (C:\path).
            #
            # is_absolute() guard: on Windows a POSIX-style "/path" (or a drive-relative
            # "C:path") normalizes to a *drive-relative* WindowsPath, and Path.as_uri()
            # raises ValueError for it. Such strings are not usable file paths on this
            # platform and continue to the relative-path/text handling below.
            #
            # _resolve_local_file_uri returns a URI only for an existing allowed file,
            # mirroring the relative-path branch below. A "/"-prefixed string that is
            # not an existing allowed file (e.g. a plain text note such as "/remember
            # to call Bob") falls through to text ingestion on every platform instead
            # of becoming a broken file:// URI.
            local_uri = _resolve_local_file_uri(data_item)
            if local_uri:
                require_server_path_access()
                return StoredFile(file_path=local_uri)
        # Data is a relative file path
        local_uri = _resolve_local_file_uri(data_item)
        if local_uri:
            require_server_path_access()
            return StoredFile(file_path=local_uri)

        # data is text, save it to data storage and return the file path
        return await save_data_to_file_detailed(data_item)

    if isinstance(data_item, DataItem):
        if data_item.literal_text and isinstance(data_item.data, str):
            # Store verbatim: never read this string as a URL, s3 path or local
            # path (see DataItem.literal_text). Every storage path ends here, so
            # the flag holds on add(), its incremental pre-save, and update().
            return await save_data_to_file_detailed(data_item.data)
        if data_item.cognee_path:
            # A path cognee produced itself (a file of a repository it cloned
            # from an admitted URL), never one the caller named: no gate.
            local_uri = _resolve_local_file_uri(data_item.data, strict=True)
            if local_uri:
                return StoredFile(file_path=local_uri)
            raise IngestionError(message="Local file does not exist or is not a file.")
        return await save_data_item_to_storage_detailed(data_item.data)

    # data is not a supported type
    raise IngestionError(message=f"Data type not supported: {type(data_item)}")


async def save_data_item_to_storage(data_item: BinaryIO | str | Any) -> str:
    """Put ``data_item`` in cognee storage and return its path.

    Thin wrapper over :func:`save_data_item_to_storage_detailed` for callers
    that do not need the metadata.
    """
    stored = await save_data_item_to_storage_detailed(data_item)

    return stored.file_path
