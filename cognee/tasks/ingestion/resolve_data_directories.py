import os
from urllib.parse import urlparse
from typing import List, Union, BinaryIO

from cognee.tasks.ingestion.exceptions import S3FileSystemNotFoundError
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.files.storage.s3_config import get_s3_config


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
    s3_config = get_s3_config()

    fs = None
    if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
        import s3fs

        fs = s3fs.S3FileSystem(
            key=s3_config.aws_access_key_id,
            secret=s3_config.aws_secret_access_key,
            token=s3_config.aws_session_token,
            anon=False,
        )

    for item in data:
        if isinstance(item, str):  # Check if the item is a path
            # S3
            if urlparse(item).scheme == "s3":
                if fs is not None:
                    if include_subdirectories:
                        base_path = item if item.endswith("/") else item + "/"
                        s3_keys = fs.glob(base_path + "**")
                        # If path is not directory attempt to add item directly
                        if not s3_keys:
                            s3_keys = fs.ls(item)
                    else:
                        s3_keys = fs.ls(item)
                    # Filter out keys that represent directories using fs.isdir
                    s3_files = []
                    for key in s3_keys:
                        if not fs.isdir(key):
                            if not key.startswith("s3://"):
                                s3_files.append("s3://" + key)
                            else:
                                s3_files.append(key)
                    resolved_data.extend(s3_files)
                else:
                    raise S3FileSystemNotFoundError()

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
