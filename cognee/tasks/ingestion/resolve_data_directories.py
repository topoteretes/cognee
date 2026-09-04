from pathlib import Path
from urllib.parse import urlparse
from typing import List, Union, BinaryIO

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
    user=None,
    dataset_id=None,
):
    """
    Resolves directories by replacing them with their contained files.

    A GitHub/GitLab repository URL (``code_repo_clone_url``) is shallow-cloned
    and then resolved exactly like a local code project directory, below.
    Other http(s) URLs pass through untouched and are fetched as web pages by
    ``save_data_item_to_storage``.

    A local directory that IS a code project (see
    ``cognee.tasks.code_graph.code_repo.PROJECT_MARKERS``) is not flattened:
    it resolves to ONE repo-level manifest DataItem (cognify runs a single
    enola pass over the whole project) plus its document-like files as
    individual items; VCS internals, caches, dotfiles, and binaries are
    skipped. ``user``/``dataset_id`` pin the repo item's stable identity so
    re-adds update one record — pass them when available (S3 paths and plain
    directories are unaffected).

    Args:
        data: A single file, directory, or binary stream, or a list of such items.
        include_subdirectories: Whether to include files in subdirectories recursively.
        user: Owner used to pin repo-manifest data ids (optional).
        dataset_id: Dataset used to pin repo-manifest data ids (optional).

    Returns:
        A list of resolved files, DataItems, and binary streams.
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
                continue

            # A GitHub/GitLab repository URL is a code project, not a web page:
            # clone it and resolve it like the local project directory below --
            # one code_repo manifest plus the repo's documents. Deferred import:
            # code_repo reaches back into this package (dlt_utils).
            from cognee.tasks.code_graph.resolve_repo import code_repo_clone_url

            if code_repo_clone_url(item) is not None:
                from cognee.tasks.code_graph.code_repo import resolve_code_repository_url

                manifest_item, document_paths, _skipped = await resolve_code_repository_url(
                    item, user=user, dataset_id=dataset_id
                )
                resolved_data.append(manifest_item)
                resolved_data.extend(str(path) for path in document_paths)
                continue

            local_path = _resolve_existing_local_path(item)

            if local_path and local_path.is_dir():  # If it's a directory
                if include_subdirectories:
                    # A code project resolves to one repo item + its documents
                    # instead of a flat file list. Deferred import: code_repo
                    # reaches back into this package (dlt_utils).
                    from cognee.tasks.code_graph.code_repo import (
                        detect_code_project,
                        resolve_code_repository,
                    )

                    if detect_code_project(local_path):
                        manifest_item, document_paths, _skipped = await resolve_code_repository(
                            local_path, user=user, dataset_id=dataset_id
                        )
                        resolved_data.append(manifest_item)
                        resolved_data.extend(str(path) for path in document_paths)
                        continue

                    # Recursively add all files in the directory and subdirectories
                    for file_path in local_path.rglob("*"):
                        if file_path.is_file():
                            resolved_data.append(str(file_path))
                else:
                    # Add all files (not subdirectories) in the directory
                    resolved_data.extend(
                        str(file_path) for file_path in local_path.iterdir() if file_path.is_file()
                    )
            elif local_path and local_path.is_file():
                resolved_data.append(str(local_path))
            else:  # If it's a file or text add it directly
                resolved_data.append(item)
        else:  # If it's not a string add it directly
            resolved_data.append(item)
    return resolved_data
