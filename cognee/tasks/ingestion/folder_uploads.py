"""Folder uploads: multipart parts whose filenames carry a relative path.

HTTP has no folder primitive -- ``multipart/form-data`` is a flat list of
parts, each with a filename. A client uploads a folder by sending every file
as its own ``data`` part and putting the path relative to the folder in the
filename (``proj/pyproject.toml``, ``proj/src/app.py``); browsers do this from
``<input webkitdirectory>`` via ``FormData.append(name, file, relativePath)``.

Two stages, deliberately split:

- ``validate_folder_uploads`` runs at the API boundary. It is pure and cheap,
  and lets a bad request fail as a 400 instead of as a pipeline error.
- ``materialize_folder_uploads`` runs inside the add pipeline, from
  ``resolve_data_directories``, after the dataset is authorized and with the
  user and dataset in hand. It groups folder parts by their first segment,
  writes each group under ``<repos_root_directory>/uploads/<user>/<dataset>/
  <folder>`` and replaces the parts with that directory, which the rest of
  ``resolve_data_directories`` then treats like any local directory: a code
  project becomes ONE ``code_repo`` manifest (cognify runs enola over it)
  plus its documents, a plain folder is flattened to its files. Parts without
  a separator are untouched, and only objects with a ``filename`` attribute
  (uploads) count -- a local file handle's ``name`` is a path on this machine,
  not a relative name.

The folder is not a temp dir: cognify's CODE_REPO route reads the original
directory when it runs, which may be a later ``/cognify`` call. Re-uploading a
folder of the same name into the same dataset replaces the copy wholesale, so
the manifest keeps its identity (keyed on the path) and content-change
detection drives the re-cognify, exactly like a refreshed git clone. Scoping
by dataset keeps two datasets' ``proj/`` folders apart.
"""

import inspect
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

from cognee.base_config import get_base_config
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.exceptions import InvalidFolderUploadError


def folder_uploads_root() -> Path:
    """Where uploaded folders are written: ``<repos_root_directory>/uploads``.

    Lives under the clones root so it is one of ingestion's always-allowed
    local file roots (local_path_safety) and is configured by the same
    ``COGNEE_REPOS_DIR``.
    """
    return Path(get_base_config().repos_root_directory) / "uploads"


def relative_upload_parts(filename: Optional[str]) -> Optional[Tuple[str, ...]]:
    """The path segments of a folder-upload filename, or None for a flat upload.

    Backslashes count as separators (Windows clients). Absolute paths and any
    empty, ``.`` or ``..`` segment are rejected: the name is client-supplied
    and decides where bytes land on the server.
    """
    if not filename:
        return None
    name = filename.replace("\\", "/")
    if "/" not in name:
        return None
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise InvalidFolderUploadError(
            message=f"Upload name '{filename}' is an absolute path; send paths relative to "
            "the uploaded folder, e.g. 'proj/src/app.py'."
        )
    parts = tuple(name.split("/"))
    if any(part in ("", ".", "..") or "\x00" in part for part in parts):
        raise InvalidFolderUploadError(
            message=f"Upload name '{filename}' contains an empty, '.' or '..' path segment."
        )
    return parts


def folder_parts_of(item: Any) -> Optional[Tuple[str, ...]]:
    """``relative_upload_parts`` for an upload-like item, None for anything else."""
    if isinstance(item, DataItem) or not hasattr(item, "filename"):
        return None
    return relative_upload_parts(getattr(item, "filename", None))


def validate_folder_uploads(uploads: Optional[list], with_attributes: bool = False) -> bool:
    """Boundary check for a request's uploads; returns whether any is a folder part.

    Raises ``InvalidFolderUploadError`` (400) for an unsafe name, and when
    ``with_attributes`` (labels / external_metadata were sent) and a folder
    part is present: a folder expands to many records inside the pipeline, so
    a positional label for it has nothing to attach to.
    """
    # A list, not a generator: every name must be checked, not just the ones
    # before the first folder part.
    is_folder_part = [folder_parts_of(upload) is not None for upload in uploads or []]
    has_folder = any(is_folder_part)
    if has_folder and with_attributes:
        raise InvalidFolderUploadError(
            message="labels and external_metadata are not supported with folder uploads — "
            "upload the folder on its own, or send its files as flat uploads."
        )
    return has_folder


async def _read_upload(upload: Any) -> bytes:
    """Bytes of an UploadFile (async read) or a plain file-like (sync read)."""
    stream = getattr(upload, "file", None)
    if stream is not None and hasattr(stream, "seek"):
        stream.seek(0)
    payload = upload.read() if hasattr(upload, "read") else stream.read()
    if inspect.isawaitable(payload):
        payload = await payload
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return payload or b""


async def materialize_folder_uploads(items: list, user, dataset_id) -> list:
    """Replace folder parts in ``items`` with the directory they were written to.

    Order is preserved: the directory takes the position of its first part and
    the other parts of that folder are dropped; everything else passes through
    untouched. Every name is validated before anything is written, so a bad
    part rejects the whole batch cleanly. A folder part wrapped in a DataItem
    (a label or metadata was attached) is rejected for the reason given in
    ``validate_folder_uploads``.
    """
    grouped: dict = {}
    for index, item in enumerate(items):
        if isinstance(item, DataItem) and folder_parts_of(item.data) is not None:
            raise InvalidFolderUploadError(
                message="labels and external_metadata are not supported with folder uploads."
            )
        parts = folder_parts_of(item)
        if parts is not None:
            grouped.setdefault(parts[0], []).append((index, parts[1:], item))

    if not grouped:
        return items
    if user is None or dataset_id is None:
        raise InvalidFolderUploadError(
            message="Folder uploads need the ingestion context (user and dataset) to be "
            "written; add() supplies it when data_cache or incremental_loading is on."
        )

    dataset_root = folder_uploads_root() / str(user.id) / str(dataset_id)
    replacements: dict = {}
    dropped: set = set()
    for folder_name, entries in grouped.items():
        folder = dataset_root / folder_name
        # Replace, never merge: the request is the whole folder as the client
        # sees it now, so a file the client deleted must not linger.
        if folder.exists():
            shutil.rmtree(folder)
        for position, (index, relative_parts, upload) in enumerate(entries):
            destination = folder.joinpath(*relative_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(await _read_upload(upload))
            if position == 0:
                replacements[index] = str(folder)
            else:
                dropped.add(index)

    resolved: List[Any] = []
    for index, item in enumerate(items):
        if index in dropped:
            continue
        resolved.append(replacements.get(index, item))
    return resolved
