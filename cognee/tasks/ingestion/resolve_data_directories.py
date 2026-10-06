from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from cognee.infrastructure.files.storage.s3_config import get_s3_config
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.exceptions import S3FileSystemNotFoundError


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
    data: BinaryIO | list[BinaryIO] | str | list[str],
    include_subdirectories: bool = True,
    user=None,
    dataset_id=None,
    credentials: str | None = None,
    include_documents: bool = True,
    treat_as_repository: bool = False,
):
    """
    Resolves directories by replacing them with their contained files.

    A GitHub/GitLab repository URL (``code_repo_clone_url``) is shallow-cloned
    and then resolved exactly like a local code project directory, below.
    Other http(s) URLs pass through untouched and are fetched as web pages by
    ``save_data_item_to_storage``.

    The three repository options are ``add()``'s ``codegraph_config``:
    ``credentials`` authenticates the clone of a private repository URL,
    ``include_documents=False`` emits a repository's manifest without its
    README/docs, and ``treat_as_repository`` takes EVERY item as a repository
    spec (``resolve_declared_repositories``) instead of detecting them. The
    pipeline runner calls this with the defaults; ``add()`` calls it ahead of
    the run when any option is set, and this function is idempotent on its
    own output, so the later calls pass the resolved items through.

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
        credentials: Token for cloning private repository URLs (optional).
        include_documents: Also emit a repository's document files (default True).
        treat_as_repository: Resolve every item as a declared repository spec.

    Returns:
        A list of resolved files, DataItems, and binary streams.
    """
    if treat_as_repository:
        # The caller has declared what these are: nothing is detected, every
        # item is cloned or used in place. Deferred import: code_repo reaches
        # back into this package (dlt_utils).
        from cognee.tasks.code_graph.code_repo import resolve_declared_repositories

        return await resolve_declared_repositories(
            data,
            credentials=credentials,
            include_documents=include_documents,
            user=user,
            dataset_id=dataset_id,
        )

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
            from cognee.tasks.code_graph.resolve_repo import (
                SSH_REPO_SPEC_MESSAGE,
                code_repo_clone_url,
                is_ssh_repo_spec,
            )

            # An ssh remote names a repository and nothing else, so falling
            # through to the text path below would store the spec string as a
            # document -- a silent wrong result. http(s) URLs are NOT checked
            # this way: most of them really are web pages.
            if is_ssh_repo_spec(item):
                raise IngestionError(message=SSH_REPO_SPEC_MESSAGE)

            if code_repo_clone_url(item) is not None:
                from cognee.tasks.code_graph.code_repo import resolve_code_repository_url

                manifest_item, documents, _skipped = await resolve_code_repository_url(
                    item,
                    user=user,
                    dataset_id=dataset_id,
                    credentials=credentials,
                    include_documents=include_documents,
                )
                resolved_data.append(manifest_item)
                resolved_data.extend(documents)
                continue

            local_path = _resolve_existing_local_path(item)

            if local_path and local_path.is_dir():  # If it's a directory
                # Checked here, not only per file at storage time: a code project
                # resolves to a manifest DataItem that never reaches that check,
                # and cognify then reads the directory in place.
                from cognee.tasks.ingestion.save_data_item_to_storage import (
                    settings as save_data_settings,
                )

                if not save_data_settings.accept_local_file_path:
                    raise IngestionError(
                        message="Local directories are not accepted "
                        "(ACCEPT_LOCAL_FILE_PATH=false). Pass a repository URL "
                        "or upload the files instead."
                    )
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
                            local_path,
                            user=user,
                            dataset_id=dataset_id,
                            include_documents=include_documents,
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
