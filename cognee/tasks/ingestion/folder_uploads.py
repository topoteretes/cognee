"""Folder uploads: multipart parts whose filenames carry a relative path.

HTTP has no folder primitive -- ``multipart/form-data`` is a flat list of
parts, each with a filename. A client uploads a folder by sending every file
as its own ``data`` part and putting the path relative to the folder in the
filename (``proj/pyproject.toml``, ``proj/src/app.py``); browsers do this from
``<input webkitdirectory>`` via ``FormData.append(name, file, relativePath)``.

The server groups such parts by their first path segment, writes each group
under ``<repos_root_directory>/uploads/<user_id>/<folder>``, and hands the
directory to ``add()`` in place of the parts. From there it is an ordinary
directory add: ``resolve_data_directories`` turns a code project into ONE
``code_repo`` manifest (cognify runs enola over it) plus its documents, and
flattens a plain folder to its files. Parts without a separator are left
exactly as they were.

The folder is not a temp dir: cognify's CODE_REPO route reads the original
directory when it runs, which may be a later ``/cognify`` call. Re-uploading a
folder of the same name replaces the copy wholesale, so the manifest keeps
its identity (keyed on the path) and content-change detection drives the
re-cognify, exactly like a refreshed git clone.
"""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from cognee.base_config import get_base_config
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


async def materialize_folder_uploads(uploads: Optional[list], user) -> Tuple[list, List[str]]:
    """Split uploads into flat parts and folder directories, writing the folders.

    Returns ``(flat_uploads, folder_paths)``: the parts without a separator,
    unchanged and in their original order, and one directory path per
    top-level folder name in first-seen order. Every name is validated before
    anything is written, so a bad part rejects the whole request cleanly.
    """
    flat: list = []
    folders: dict = {}
    for upload in uploads or []:
        parts = relative_upload_parts(getattr(upload, "filename", None))
        if parts is None:
            flat.append(upload)
            continue
        folders.setdefault(parts[0], []).append((parts[1:], upload))

    if not folders:
        return flat, []

    user_root = folder_uploads_root() / str(user.id)
    folder_paths: List[str] = []
    for folder_name, entries in folders.items():
        folder = user_root / folder_name
        # Replace, never merge: the request is the whole folder as the client
        # sees it now, so a file the client deleted must not linger.
        if folder.exists():
            shutil.rmtree(folder)
        for relative_parts, upload in entries:
            destination = folder.joinpath(*relative_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(await upload.read())
        folder_paths.append(str(folder))
    return flat, folder_paths
