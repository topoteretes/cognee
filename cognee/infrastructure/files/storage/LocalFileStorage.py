import os
import shutil
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import BinaryIO, Optional, Union

from .FileBufferedReader import FileBufferedReader
from .storage import Storage


def get_parsed_path(file_path: str) -> str:
    # Check if this is actually a URL (has a scheme like file://, http://, etc.)
    if "://" in file_path:
        parsed_url = urlparse(file_path)

        # Handle file:// URLs specially
        if parsed_url.scheme == "file":
            # On Windows, urlparse handles drive letters correctly
            # Convert the path component to a proper file path
            if os.name == "nt":  # Windows
                # Remove leading slash from Windows paths like /C:/Users/...
                # but handle UNC paths like //server/share correctly
                parsed_path = parsed_url.path
                if parsed_path.startswith("/") and len(parsed_path) > 1 and parsed_path[2] == ":":
                    # This is a Windows drive path like /C:/Users/...
                    parsed_path = parsed_path[1:]
                elif parsed_path.startswith("///"):
                    # This is a UNC path like ///server/share, convert to //server/share
                    parsed_path = parsed_path[1:]
            else:  # Unix-like systems
                parsed_path = parsed_url.path
        else:
            # For non-file URLs, use the path as-is
            parsed_path = parsed_url.path
            if (
                os.name == "nt"
                and parsed_path.startswith("/")
                and len(parsed_path) > 1
                and parsed_path[2] == ":"
            ):
                parsed_path = parsed_path[1:]

        # Normalize path separators to ensure consistency
        return os.path.normpath(parsed_path)
    else:
        # This is a regular file path, not a URL - normalize separators
        return os.path.normpath(file_path)


class LocalFileStorage(Storage):
    """
    Manage local file storage operations such as storing, retrieving, and managing files on
    the filesystem.
    """

    storage_path: Optional[str] = None

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def store(self, file_path: str, data: Union[BinaryIO, str], overwrite: bool = False) -> str:
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
        parsed_storage_path = get_parsed_path(self.storage_path)
        full_file_path = os.path.join(parsed_storage_path, file_path)
        file_dir_path = os.path.dirname(full_file_path)

        self.ensure_directory_exists(file_dir_path)

        if overwrite or not os.path.exists(full_file_path):
            if isinstance(data, str):
                with open(full_file_path, mode="w", encoding="utf-8", newline="\n") as file:
                    file.write(data)
            else:
                with open(full_file_path, mode="wb") as file:
                    if hasattr(data, "read"):
                        data.seek(0)
                        file.write(data.read())
                    else:
                        file.write(data)

                file.close()

        return "file://" + full_file_path

    @contextmanager
    def open(self, file_path: str, mode: str = "rb", *args, **kwargs):
        """
        Retrieve data from a specified file path, returning the content as bytes.

        This method opens the file in read mode and reads its content. The function expects the
        file to exist; if it does not, a FileNotFoundError will be raised.

        Parameters:
        -----------

            - file_path (str): The relative path of the file to retrieve data from.
            - mode (str): The mode to open the file, with "rb" as the default for reading binary
              files. (default "rb")

        Returns:
        --------

            The content of the retrieved file as bytes.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)

        full_file_path = os.path.join(parsed_storage_path, file_path)

        # Add debug information for Windows path issues
        if not os.path.exists(full_file_path):
            # Try to provide helpful debug information
            if os.path.exists(parsed_storage_path):
                available_files = []
                try:
                    available_files = os.listdir(parsed_storage_path)
                except (OSError, PermissionError):
                    available_files = ["<unable to list directory>"]

                raise FileNotFoundError(
                    f"File not found: '{full_file_path}'\n"
                    f"Storage path: '{parsed_storage_path}'\n"
                    f"Requested file: '{file_path}'\n"
                    f"Storage path exists: {os.path.exists(parsed_storage_path)}\n"
                    f"Available files in storage: {available_files[:10]}..."  # Limit to first 10 files
                )
            else:
                raise FileNotFoundError(
                    f"Storage directory does not exist: '{parsed_storage_path}'\n"
                    f"Original storage path: '{self.storage_path}'\n"
                    f"Requested file: '{file_path}'"
                )

        with open(full_file_path, mode=mode, *args, **kwargs) as file:
            file = FileBufferedReader(file, name="file://" + full_file_path)

            try:
                yield file
            finally:
                file.close()

    def file_exists(self, file_path: str):
        """
        Check if a specified file exists in the storage.

        Parameters:
        -----------

            - file_path (str): The path of the file to check for existence.

        Returns:
        --------

            - bool: True if the file exists, otherwise False.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)

        return os.path.exists(os.path.join(parsed_storage_path, file_path))

    def is_file(self, file_path: str):
        """
        Check if a specified file is a regular file.

        Parameters:
        -----------

            - file_path (str): The path of the file to check.

        Returns:
        --------

            - bool: True if the file is a regular file, otherwise False.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)

        return os.path.isfile(os.path.join(parsed_storage_path, file_path))

    def get_size(self, file_path: str) -> int:
        parsed_storage_path = get_parsed_path(self.storage_path)

        return (
            os.path.getsize(os.path.join(parsed_storage_path, file_path))
            if self.file_exists(file_path)
            else 0
        )

    def ensure_directory_exists(self, directory_path: str = ""):
        """
        Ensure that the specified directory exists, creating it if necessary.

        If the directory already exists, no action is taken.

        Parameters:
        -----------

            - directory_path (str): The path of the directory to check or create.
        """
        if not directory_path.strip():
            directory_path = get_parsed_path(self.storage_path)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

    def copy_file(self, source_file_path: str, destination_file_path: str):
        """
        Copy a file from a source path to a destination path.
        Files need to be in the same storage.

        Parameters:
        -----------

            - source_file_path (str): The path of the file to be copied.
            - destination_file_path (str): The path where the file will be copied to.

        Returns:
        --------

            - str: The path to the copied file.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)

        return shutil.copy2(
            os.path.join(parsed_storage_path, source_file_path),
            os.path.join(parsed_storage_path, destination_file_path),
        )

    def remove(self, file_path: str):
        """
        Remove the specified file from the storage if it exists.

        Parameters:
        -----------

            - file_path (str): The path of the file to be removed.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)
        full_file_path = os.path.join(parsed_storage_path, file_path)

        if os.path.exists(full_file_path):
            os.remove(full_file_path)

    def list_files(self, directory_path: str, recursive: bool = False) -> list[str]:
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
        from pathlib import Path

        parsed_storage_path = get_parsed_path(self.storage_path)

        if directory_path:
            full_directory_path = os.path.join(parsed_storage_path, directory_path)
        else:
            full_directory_path = parsed_storage_path

        directory_pathlib = Path(full_directory_path)

        if not directory_pathlib.exists() or not directory_pathlib.is_dir():
            return []

        files = []

        if recursive:
            # Use rglob for recursive search
            for file_path in directory_pathlib.rglob("*"):
                if file_path.is_file():
                    # Get relative path from storage root
                    relative_path = os.path.relpath(str(file_path), parsed_storage_path)
                    # Normalize path separators for consistency
                    relative_path = relative_path.replace(os.sep, "/")
                    files.append(relative_path)
        else:
            # Use iterdir for just immediate directory
            for file_path in directory_pathlib.iterdir():
                if file_path.is_file():
                    # Get relative path from storage root
                    relative_path = os.path.relpath(str(file_path), parsed_storage_path)
                    # Normalize path separators for consistency
                    relative_path = relative_path.replace(os.sep, "/")
                    files.append(relative_path)

        return files

    def remove_all(self, tree_path: str = None):
        """
        Remove an entire directory tree at the specified path, including all files and
        subdirectories.

        If the directory does not exist, no action is taken and no exception is raised.

        If directories don't exist in the storage we ignore it.

        Parameters:
        -----------

            - tree_path (str): The root path of the directory tree to be removed.
        """
        parsed_storage_path = get_parsed_path(self.storage_path)

        if tree_path is None:
            tree_path = parsed_storage_path
        else:
            tree_path = os.path.join(parsed_storage_path, tree_path)

        try:
            return shutil.rmtree(tree_path)
        except FileNotFoundError:
            pass
