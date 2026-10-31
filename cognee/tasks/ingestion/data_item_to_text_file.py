import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.infrastructure.files.encryption import is_encrypted_local_file
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.loaders import get_loader_engine
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.utils.run_async import run_async
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class SaveDataSettings(BaseSettings):
    accept_local_file_path: bool = True

    model_config = SettingsConfigDict(extra="allow")


settings = SaveDataSettings()


# Bytes copied per iteration when pulling a stored object down to a temp file.
# The previous 8 KiB read meant one Python-level round trip through the storage
# stack per 8 KiB of payload — ~12k iterations for a 100 MB object.
DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024


def _copy_stream(source_file, destination_file) -> None:
    """Copy a stream in chunks. Blocking: callers run this off the event loop."""
    while chunk := source_file.read(DOWNLOAD_CHUNK_SIZE):
        destination_file.write(chunk)


async def pull_to_local_copy(file_path, destination_file) -> None:
    """Copy a stored object into ``destination_file`` as the plaintext a loader can read.

    Used for every object a loader cannot open by path itself: an S3 object,
    and a locally stored file sealed by the encryption layer (``open_data_file``
    unseals it; the copy is what the loader gets).
    """
    async with open_data_file(file_path) as file:
        # Both sides of this copy block: the reads wait on the network and the
        # writes wait on the disk. Inline, it parked the event loop for the whole
        # download, so the pipeline's per-item concurrency could not overlap.
        await run_async(_copy_stream, file, destination_file)


async def _load_from_local_copy(
    data_item_path: str,
    suffix: str,
    preferred_loaders: dict[str, dict[str, Any]] | None,
    **loader_kwargs: Any,
) -> tuple[str, LoaderInterface]:
    """Run the loader engine on a temporary plaintext copy of ``data_item_path``.

    The copy is created with the process's private temp-file permissions
    (0600) and removed when the loader returns, success or not. It is the one
    place a sealed object exists unsealed outside process memory, for as long
    as the loader takes — loaders open their input by path, so there is no
    stream-only route through them today.
    """
    # TODO: Rework this to work with file streams and not saving data to temp storage
    # Note: proper suffix information is needed for OpenAI to handle mp3 files
    #
    # NamedTemporaryFile defaults to delete=True, which on Windows keeps an
    # exclusive lock on the file and forbids reopening it by name while the
    # handle is still open. The loader below reopens the file by name, so we
    # create it with delete=False, close our handle first, and clean it up
    # ourselves. (Mirrors the delete=False pattern used by the SQLAlchemy and
    # ladybug S3 temp-file paths.)
    temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed below after the write; delete=False keeps the path
        mode="wb", suffix=suffix, delete=False
    )
    try:
        await pull_to_local_copy(data_item_path, temp_file)
        temp_file.flush()  # Data needs to be saved to local storage
        temp_file.close()  # release the handle so the loader can reopen by name
        loader = get_loader_engine()
        return await loader.load_file(
            temp_file.name, preferred_loaders, **loader_kwargs
        ), loader.get_loader(temp_file.name, preferred_loaders)
    finally:
        temp_file.close()  # idempotent; covers the early-exception path
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass


async def data_item_to_text_file(
    data_item_path: str,
    preferred_loaders: dict[str, dict[str, Any]] | None = None,
    **loader_kwargs: Any,
) -> tuple[str, LoaderInterface]:
    """Run the loader engine on a data item's file.

    ``loader_kwargs`` (ingestion context: dataset_name, dataset_id, user,
    original_file_name) is forwarded to the selected loader's ``load()``;
    loaders that don't need it ignore it via ``**kwargs``.
    """
    if isinstance(data_item_path, str):
        parsed_url = urlparse(data_item_path)

        # data is s3 file path
        if parsed_url.scheme == "s3":
            return await _load_from_local_copy(
                data_item_path, Path(parsed_url.path).suffix, preferred_loaders, **loader_kwargs
            )

        # data is local file path
        elif parsed_url.scheme == "file":
            if settings.accept_local_file_path:
                if is_encrypted_local_file(get_data_file_path(data_item_path)):
                    return await _load_from_local_copy(
                        data_item_path,
                        Path(parsed_url.path).suffix,
                        preferred_loaders,
                        **loader_kwargs,
                    )
                loader = get_loader_engine()
                return await loader.load_file(
                    data_item_path, preferred_loaders, **loader_kwargs
                ), loader.get_loader(data_item_path, preferred_loaders)
            else:
                raise IngestionError(message="Local files are not accepted.")

        # data is an absolute file path
        elif data_item_path.startswith("/") or (
            os.name == "nt" and len(data_item_path) > 1 and data_item_path[1] == ":"
        ):
            # Handle both Unix absolute paths (/path) and Windows absolute paths (C:\path)
            if settings.accept_local_file_path:
                if is_encrypted_local_file(data_item_path):
                    # A sealed object under the data root: loaders read by path,
                    # so they get a temporary plaintext copy, as S3 objects do.
                    return await _load_from_local_copy(
                        data_item_path,
                        Path(data_item_path).suffix,
                        preferred_loaders,
                        **loader_kwargs,
                    )
                loader = get_loader_engine()
                return await loader.load_file(
                    data_item_path, preferred_loaders, **loader_kwargs
                ), loader.get_loader(data_item_path, preferred_loaders)
            else:
                raise IngestionError(message="Local files are not accepted.")
    # data is not a supported type
    raise IngestionError(message=f"Data type not supported: {type(data_item_path)}")
