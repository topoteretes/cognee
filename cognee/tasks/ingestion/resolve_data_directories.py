from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional, Union, BinaryIO

from cognee.tasks.ingestion.directory_file_filters import filter_directory_files, filter_s3_keys
from cognee.tasks.ingestion.exceptions import S3FileSystemNotFoundError
from cognee.infrastructure.files.storage.s3_config import get_s3_config
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path


def _resolve_existing_local_path(item: str) -> Path | None:
    try:
        return resolve_local_path(item, must_exist=True)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    except ValueError:
        # A path-looking string outside the allowed roots is never expanded or read
        # here; it is passed through unchanged and handled downstream by
        # save_data_item_to_storage (which ingests it as plain text).
        return None


async def resolve_data_directories(
    data: Union[BinaryIO, List[BinaryIO], str, List[str]],
    include_subdirectories: bool = True,
    respect_gitignore: bool = False,
    exclude_patterns: Optional[List[str]] = None,
):
    """
    Resolves directories by replacing them with their contained files.

    Expanded directories are filtered: version-control internals (.git and
    friends) and binary files no registered loader supports are always
    skipped, ``respect_gitignore=True`` additionally skips matches of the
    directory's top-level .gitignore, and ``exclude_patterns`` skips
    gitignore-style matches (e.g. ``*.log``, ``.venv/``). Explicitly passed
    files are never filtered. S3 prefixes get pattern filtering only —
    .gitignore and binary detection would require remote reads.

    Args:
        data: A single file, directory, or binary stream, or a list of such items.
        include_subdirectories: Whether to include files in subdirectories recursively.
        respect_gitignore: Skip files matched by the directory's top-level .gitignore.
        exclude_patterns: Gitignore-style patterns to skip when expanding directories.

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
                    resolved_data.extend(filter_s3_keys(item, s3_files, exclude_patterns))
                else:
                    raise S3FileSystemNotFoundError()
                continue

            local_path = _resolve_existing_local_path(item)

            if local_path and local_path.is_dir():  # If it's a directory
                if include_subdirectories:
                    candidates = [
                        file_path for file_path in local_path.rglob("*") if file_path.is_file()
                    ]
                else:
                    candidates = [
                        file_path for file_path in local_path.iterdir() if file_path.is_file()
                    ]
                resolved_data.extend(
                    str(file_path)
                    for file_path in filter_directory_files(
                        local_path,
                        candidates,
                        respect_gitignore=respect_gitignore,
                        exclude_patterns=exclude_patterns,
                    )
                )
            elif local_path and local_path.is_file():
                resolved_data.append(str(local_path))
            else:  # If it's a file or text add it directly
                resolved_data.append(item)
        else:  # If it's not a string add it directly
            resolved_data.append(item)
    return resolved_data
