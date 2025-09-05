import os
from typing import List, Union, BinaryIO

from cognee.tasks.ingestion.exceptions import CloudFileSystemNotFoundError
from cognee.infrastructure.files.storage import StorageProviderRegistry
from cognee.infrastructure.files.storage.utils import get_scheme_with_separator


async def resolve_data_directories(
    data: Union[BinaryIO, List[BinaryIO], str, List[str]], include_subdirectories: bool = True
):
    """
    Resolves directories by replacing them with their contained files.

    Args:
        data: A single file, directory, or binary stream, or a list of such items.
        include_subdirectories: Whether to include files in subdirectories recursively.

    Returns:
        A list of resolved files and binary streams.
    """
    # Ensure `data` is a list
    if not isinstance(data, list):
        data = [data]

    resolved_data = []

    for item in data:
        if isinstance(item, str):  # Check if the item is a path
            scheme_with_separator = get_scheme_with_separator(item)
            # Check if the item is a cloud storage(s3, gcs, azure, etc.) file path
            if scheme_with_separator in StorageProviderRegistry.get_all_cloud_schemes():
                cloud_storage_cls = StorageProviderRegistry.get_provider_by_cloud_scheme(
                    scheme_with_separator
                )
                cloud_storage = cloud_storage_cls(scheme_with_separator)
                fs = cloud_storage.fs

                if fs is not None:
                    if include_subdirectories and fs.isdir(item):
                        base_path = item if item.endswith("/") else item + "/"
                        keys = fs.glob(base_path + "**")
                    else:
                        keys = fs.ls(item)
                    # Filter out keys that represent directories using fs.isdir
                    files = []
                    for key in keys:
                        if not fs.isdir(key):
                            if not key.startswith(scheme_with_separator):
                                files.append(scheme_with_separator + key)
                            else:
                                files.append(key)
                    resolved_data.extend(files)
                else:
                    raise CloudFileSystemNotFoundError()

            elif os.path.isdir(item):  # If it's a directory
                if include_subdirectories:
                    # Recursively add all files in the directory and subdirectories
                    for root, _, files in os.walk(item):
                        resolved_data.extend([os.path.join(root, f) for f in files])
                else:
                    # Add all files (not subdirectories) in the directory
                    resolved_data.extend(
                        [
                            os.path.join(item, f)
                            for f in os.listdir(item)
                            if os.path.isfile(os.path.join(item, f))
                        ]
                    )
            else:  # If it's a file or text add it directly
                resolved_data.append(item)
        else:  # If it's not a string add it directly
            resolved_data.append(item)
    return resolved_data
