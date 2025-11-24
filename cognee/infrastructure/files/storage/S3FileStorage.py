import os
from typing import BinaryIO, Union, TYPE_CHECKING
from contextlib import asynccontextmanager

from cognee.infrastructure.files.storage.s3_config import get_s3_config
from cognee.infrastructure.utils.run_async import run_async
from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader
from .storage import Storage

if TYPE_CHECKING:
    import s3fs


class S3FileStorage(Storage):
    """
    Manage S3 file storage operations such as storing, retrieving, and managing files on
    S3-compatible storage.
    """

    storage_path: str
    s3: "s3fs.S3FileSystem"

    def __init__(self, storage_path: str):
        try:
            import s3fs
        except ImportError:
            raise ImportError(
                's3fs is required for S3FileStorage. Install it with: pip install cognee"[aws]"'
            )

        self.storage_path = storage_path
        s3_config = get_s3_config()
        if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
            self.s3 = s3fs.S3FileSystem(
                key=s3_config.aws_access_key_id,
                secret=s3_config.aws_secret_access_key,
                token=s3_config.aws_session_token,
                anon=False,
                endpoint_url=s3_config.aws_endpoint_url,
                client_kwargs={"region_name": s3_config.aws_region},
            )
        else:
            raise ValueError("S3 credentials are not set in the configuration.")

    async def store(
        self, file_path: str, data: Union[BinaryIO, str], overwrite: bool = False
    ) -> str:
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
        full_file_path = os.path.join(self.storage_path.replace("s3://", ""), file_path)

        file_dir_path = os.path.dirname(full_file_path)

        await self.ensure_directory_exists(file_dir_path)

        if overwrite or not await self.file_exists(file_path):

            def save_data_to_file():
                if isinstance(data, str):
                    with self.s3.open(
                        full_file_path, mode="w", encoding="utf-8", newline="\n"
                    ) as file:
                        file.write(data)
                else:
                    with self.s3.open(full_file_path, mode="wb") as file:
                        if hasattr(data, "read"):
                            data.seek(0)
                            file.write(data.read())
                        else:
                            file.write(data)

            await run_async(save_data_to_file)

        return "s3://" + full_file_path

    @asynccontextmanager
    async def open(self, file_path: str, mode: str = "r"):
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
        full_file_path = os.path.join(self.storage_path.replace("s3://", ""), file_path)

        def get_file():
            return self.s3.open(full_file_path, mode=mode)

        file = await run_async(get_file)
        file = FileBufferedReader(file, name="s3://" + full_file_path)

        try:
            yield file
        finally:
            file.close()

    async def file_exists(self, file_path: str):
        """
        Check if a specified file exists in the filesystem.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        return await run_async(
            self.s3.exists, os.path.join(self.storage_path.replace("s3://", ""), file_path)
        )

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
        return await run_async(
            self.s3.isfile, os.path.join(self.storage_path.replace("s3://", ""), file_path)
        )

    async def get_size(self, file_path: str) -> int:
        return await run_async(
            self.s3.size, os.path.join(self.storage_path.replace("s3://", ""), file_path)
        )

    async def ensure_directory_exists(self, directory_path: str = ""):
        """
        Ensure that the specified directory exists, creating it if necessary.

        For S3 storage, this is a no-op since directories are created implicitly
        when files are written to paths. S3 doesn't have actual directories,
        just object keys with prefixes that appear as directories.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        # In S3, directories don't exist as separate entities - they're just prefixes
        # When you write a file to s3://bucket/path/to/file.txt, the "directories"
        # path/ and path/to/ are implicitly created. No explicit action needed.
        pass

    async def copy_file(self, source_file_path: str, destination_file_path: str):
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
            return self.s3.copy(
                os.path.join(self.storage_path.replace("s3://", ""), source_file_path),
                os.path.join(self.storage_path.replace("s3://", ""), destination_file_path),
                recursive=True,
            )

        return await run_async(copy)

    async def remove(self, file_path: str):
        """
        Remove the specified file from the filesystem if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        full_file_path = os.path.join(self.storage_path.replace("s3://", ""), file_path)

        def remove_file():
            if self.s3.exists(full_file_path):
                self.s3.rm_file(full_file_path)

        await run_async(remove_file)

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
                # Combine storage path with directory path
                full_path = os.path.join(self.storage_path.replace("s3://", ""), directory_path)
            else:
                full_path = self.storage_path.replace("s3://", "")

            if recursive:
                # Use ** for recursive search
                pattern = f"{full_path}/**"
            else:
                # Just files in the immediate directory
                pattern = f"{full_path}/*"

            # Use s3fs glob to find files
            try:
                all_paths = self.s3.glob(pattern)
                # Filter to only files (not directories)
                files = [path for path in all_paths if self.s3.isfile(path)]

                # Convert back to relative paths from storage root
                storage_prefix = self.storage_path.replace("s3://", "")
                relative_files = []
                for file_path in files:
                    if file_path.startswith(storage_prefix):
                        relative_path = file_path[len(storage_prefix) :].lstrip("/")
                        relative_files.append(relative_path)

                return relative_files
            except Exception:
                # If directory doesn't exist or other error, return empty list
                return []

        return await run_async(list_files_sync)

    async def remove_all(self, tree_path: str):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        if tree_path is None:
            tree_path = self.storage_path.replace("s3://", "")
        else:
            tree_path = os.path.join(self.storage_path.replace("s3://", ""), tree_path)

        # async_remove_all = run_async(lambda: self.s3.rm(tree_path, recursive=True))

        try:
            # await async_remove_all()
            await run_async(self.s3.rm, tree_path, recursive=True)
        except FileNotFoundError:
            pass
