import posixpath
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar

from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader
from cognee.infrastructure.utils.run_async import run_async
from cognee.shared.logging_utils import get_logger

from .storage import Storage

logger = get_logger(__name__)

# Threshold in seconds above which remote storage operations are logged as slow
SLOW_OPERATION_THRESHOLD_SEC = 30.0

# Bytes read from a source stream per write iteration — the peak RAM one
# upload adds, independent of file size.
WRITE_CHUNK_SIZE = 4 * 1024 * 1024

T = TypeVar("T")

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


class RemoteFileStorage(Storage):
    """
    File storage on an object store reached through an fsspec filesystem.

    A root is ``<scheme>://<bucket-or-container>/<prefix>``. Subclasses set ``scheme``
    and build the filesystem in ``_create_filesystem``. Every filesystem call goes
    through ``_call``, so a subclass can turn its provider's errors into clearer ones.
    """

    scheme: str
    storage_path: str
    fs: "AbstractFileSystem"

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.fs = self._create_filesystem()

    def _create_filesystem(self) -> "AbstractFileSystem":
        """Build the backend's fsspec filesystem from its settings."""
        raise NotImplementedError

    def _root(self) -> str:
        """The storage root as a filesystem path, without its scheme."""
        return self.storage_path.replace(f"{self.scheme}://", "")

    def _full_path(self, file_path: str) -> str:
        return posixpath.join(self._root(), file_path)

    def _uri(self, full_path: str) -> str:
        return f"{self.scheme}://{full_path}"

    def _translate_error(self, error: Exception) -> Exception:
        """The exception to raise for a failed filesystem call; the error itself by default."""
        return error

    async def _call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking filesystem call off the event loop."""
        try:
            return await run_async(func, *args, **kwargs)
        except Exception as error:
            translated = self._translate_error(error)
            if translated is error:
                raise
            raise translated from error

    async def store(self, file_path: str, data: BinaryIO | str, overwrite: bool = False) -> str:
        """
        Store data into a specified file path. The data can be either a string or a binary
        stream.

        This method ensures that the storage directory exists before attempting to write the
        data. If the provided data is a stream, it reads from the stream and writes to the file;
        otherwise, it directly writes the provided data.

        Parameters:
        -----------

            - file_path (str): The relative path of the file where the data will be stored.
            - data (Union[BinaryIO, str]): The data to be stored, which can be a string or a
              binary stream.
            - overwrite (bool): If True, overwrite the existing file.
        """
        full_file_path = self._full_path(file_path)

        file_dir_path = posixpath.dirname(full_file_path)

        await self.ensure_directory_exists(file_dir_path)

        if overwrite or not await self.file_exists(file_path):

            def save_data_to_file():
                if isinstance(data, str):
                    with self.fs.open(
                        full_file_path, mode="w", encoding="utf-8", newline="\n"
                    ) as file:
                        file.write(data)
                else:
                    with self.fs.open(full_file_path, mode="wb") as file:
                        if hasattr(data, "read"):
                            # Chunked copy: large uploads arrive as spooled
                            # temp files on disk, and a single read() would
                            # materialize the whole payload in RAM. fsspec
                            # object stores flush a part (S3 multipart, Azure
                            # block) whenever the buffer crosses the block
                            # size, so peak memory is max(chunk, block).
                            data.seek(0)
                            while chunk := data.read(WRITE_CHUNK_SIZE):
                                file.write(chunk)
                        else:
                            file.write(data)

            await self._call(save_data_to_file)

        return self._uri(full_file_path)

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "r") -> AsyncGenerator[FileBufferedReader]:
        """
        Retrieve data from a specified file path, returning the content as bytes.

        This method opens the file in read mode and reads its content. The function expects the
        file to exist; if it does not, a FileNotFoundError will be raised.

        Parameters:
        -----------

            - file_path (str): The relative path of the file to retrieve data from.
            - mode (str): The mode to open the file, with "r" as the default for reading binary
              files. (default "r")

        Returns:
        --------

            The content of the retrieved file as bytes.
        """
        full_file_path = self._full_path(file_path)
        start_time = time.perf_counter()

        logger.debug(
            "Opening remote file",
            extra={
                "storage_path": self.storage_path,
                "file_path": file_path,
            },
        )

        def get_file():
            return self.fs.open(full_file_path, mode=mode)

        file = await self._call(get_file)
        file = FileBufferedReader(file, name=self._uri(full_file_path))

        open_elapsed = time.perf_counter() - start_time
        duration_sec = round(open_elapsed, 3)
        if open_elapsed > SLOW_OPERATION_THRESHOLD_SEC:
            logger.warning(
                "Remote file open slow",
                extra={
                    "storage_path": self.storage_path,
                    "file_path": file_path,
                    "duration_seconds": duration_sec,
                    "threshold_seconds": SLOW_OPERATION_THRESHOLD_SEC,
                },
            )
        else:
            logger.info(
                "Opened remote file",
                extra={
                    "storage_path": self.storage_path,
                    "file_path": file_path,
                    "open_duration_seconds": duration_sec,
                },
            )

        try:
            yield file
        finally:
            file.close()
            total_elapsed = time.perf_counter() - start_time
            logger.debug(
                "Closed remote file",
                extra={
                    "storage_path": self.storage_path,
                    "file_path": file_path,
                    "total_duration_seconds": round(total_elapsed, 3),
                },
            )

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a specified file exists in the filesystem.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return await self._call(self.fs.exists, self._full_path(file_path))

    async def is_file(self, file_path: str) -> bool:
        """
        Check if a specified file is a regular file.

        Parameters:
        -----------

            - file_path (str): The path of the file to check.

        Returns:
        --------

            - bool: True if the file is a regular file, otherwise False.
        """
        return await self._call(self.fs.isfile, self._full_path(file_path))

    async def get_size(self, file_path: str) -> int:
        return await self._call(self.fs.size, self._full_path(file_path))

    async def download_to(self, local_path: str) -> None:
        """Copy the object (or tree) at the storage root to ``local_path``.

        For embedded databases kept on the object store. Raises FileNotFoundError
        when nothing is stored there yet.
        """
        await self._call(self.fs.get, self._root(), local_path, recursive=True)

    async def upload_from(self, local_path: str) -> None:
        """Copy the local file (or tree) at ``local_path`` to the storage root."""
        await self._call(self.fs.put, local_path, self._root(), recursive=True)

    async def ensure_directory_exists(self, directory_path: str = "") -> None:
        """
        Ensure that the specified directory exists, creating it if necessary.

        A no-op: object stores (S3, Azure Blob) have no real directories, only
        object keys whose prefixes appear as directories once a file is written.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """

    async def copy_file(self, source_file_path: str, destination_file_path: str) -> str:
        """
        Copy a file from a source path to a destination path.

        Parameters:
        -----------

            - source_file_path (str): The path of the file to be copied.
            - destination_file_path (str): The path where the file will be copied to.

        Returns:
        --------

            - str: The path to the copied file.
        """

        def copy():
            return self.fs.copy(
                self._full_path(source_file_path),
                self._full_path(destination_file_path),
                recursive=True,
            )

        return await self._call(copy)

    async def remove(self, file_path: str) -> None:
        """
        Remove the specified file from the filesystem if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        full_file_path = self._full_path(file_path)

        def remove_file():
            if self.fs.exists(full_file_path):
                self.fs.rm_file(full_file_path)

        await self._call(remove_file)

    async def list_files(self, directory_path: str, recursive: bool = False) -> list[str]:
        """
        List all files in the specified directory.

        Parameters:
        -----------
            - directory_path (str): The directory path to list files from
            - recursive (bool): If True, list files recursively in subdirectories

        Returns:
        --------
            - list[str]: List of file paths relative to the storage root
        """

        def list_files_sync():
            if directory_path:
                full_path = self._full_path(directory_path)
            else:
                full_path = self._root()

            if recursive:
                # Use ** for recursive search
                pattern = f"{full_path}/**"
            else:
                # Just files in the immediate directory
                pattern = f"{full_path}/*"

            # Only a missing directory means "no files"; permission, transport and
            # endpoint errors propagate, as they do for LocalFileStorage, so a
            # failed listing is never mistaken for an empty one.
            try:
                all_paths = self.fs.glob(pattern)
                # Filter to only files (not directories)
                files = [path for path in all_paths if self.fs.isfile(path)]
            except FileNotFoundError:
                logger.debug(
                    "Directory not found in RemoteFileStorage.list_files",
                    exc_info=True,
                )
                return []

            # Convert back to relative paths from storage root
            storage_prefix = self._root()
            relative_files = []
            for file_path in files:
                if file_path.startswith(storage_prefix):
                    relative_path = file_path[len(storage_prefix) :].lstrip("/")
                    relative_files.append(relative_path)

            return relative_files

        return await self._call(list_files_sync)

    async def remove_all(self, root_path: str | None = None) -> None:
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        if root_path is None:
            root_path = self._root()
        else:
            root_path = self._full_path(root_path)

        try:
            await self._call(self.fs.rm, root_path, recursive=True)
        except FileNotFoundError:
            pass
