"""Refuse an add() of a file the dataset already holds with other content.

add() creates documents and leaves existing ones alone. A file that matches a
stored document by origin but not by content is an update, and updates go
through update() so the document keeps its id and its graph is replaced in
place instead of a second copy being minted. Identical content is not an
error: ingestion deduplicates it by content hash and the re-add is a no-op.

This runs once, over the whole request, before the add pipeline starts: the
pipeline processes items one at a time, so a check inside it would refuse the
offending file only after the files before it were written.

Origin is the source path for a local file (``Data.original_data_location``)
and the filename for an upload, whose stored location carries a content hash
and so cannot be matched. Raw text is named by its content hash: a changed
text never matches and is a new document by construction. Pinned items (DLT
manifests, code-repo manifests, update()'s own re-add) name their document
and are exempt. The lookup shares the dedup scope, (dataset, owner, tenant),
so another user's same-named file is never a conflict.

Not covered: s3:// objects and repository URLs, whose bytes are not local.
"""

import os
from pathlib import Path, PureWindowsPath
from typing import Any

from sqlalchemy import or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path
from cognee.modules.data.models.Data import Data
from cognee.modules.users.models import User
from cognee.tasks.ingestion.data_item import DataItem


def _local_file(item: Any) -> Path | None:
    """The existing local file ``item`` names, or None for anything else."""
    if not isinstance(item, (str, Path)):
        return None
    try:
        path = resolve_local_path(item, must_exist=True)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return path if path.is_file() else None


def _local_files(item: Any) -> list[Path]:
    """``item`` as local files: itself, or every file under it when it is a directory."""
    if isinstance(item, (str, Path)):
        try:
            path = resolve_local_path(item, must_exist=True)
        except (FileNotFoundError, OSError, ValueError):
            return []
        if path.is_dir():
            return sorted(p for p in path.rglob("*") if p.is_file())
    single = _local_file(item)
    return [single] if single else []


async def _origins(data: Any) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Origin keys of the unpinned inputs, each mapped to its content hash."""
    items = data if isinstance(data, list) else [data]
    by_location: dict[str, str] = {}
    by_name: dict[tuple[str, str], str] = {}
    for item in items:
        if isinstance(item, DataItem):
            if item.data_id is not None:
                continue
            item = item.data
        for path in _local_files(item):
            with open(path, "rb") as file:
                metadata = await get_file_metadata(file, name=path.name)
            by_location[path.as_uri()] = metadata["content_hash"]
        stream = getattr(item, "file", None)
        filename = getattr(item, "filename", None)
        if stream is not None and filename:
            # An upload is stored under its user-visible filename: the stem is
            # the row's name (ingestion derives it the same way), the extension
            # comes from the content type guessed with the filename as a hint.
            metadata = await get_file_metadata(stream, name=str(filename))
            stream.seek(0)
            stem = PureWindowsPath(str(filename)).stem
            by_name[(stem, metadata["extension"])] = metadata["content_hash"]
    return by_location, by_name


async def refuse_changed_existing_documents(data: Any, user: User, dataset) -> None:
    """Raise ``DocumentUpdateRequiredError`` for every input that is an update in disguise."""
    by_location, by_name = await _origins(data)
    if not by_location and not by_name:
        return

    tenant_filter = Data.tenant_id == user.tenant_id if user.tenant_id else Data.tenant_id.is_(None)
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(
                    Data.id,
                    Data.name,
                    Data.extension,
                    Data.original_data_location,
                    Data.content_hash,
                ).filter(
                    Data.dataset_id == dataset.id,
                    Data.owner_id == user.id,
                    tenant_filter,
                    or_(
                        Data.original_data_location.in_(list(by_location)),
                        Data.name.in_([name for name, _ in by_name]),
                    ),
                )
            )
        ).fetchall()

    conflicts: list[dict] = []
    for data_id, name, extension, original_location, stored_hash in rows:
        if original_location in by_location:
            if by_location[original_location] != str(stored_hash):
                conflicts.append(
                    {"name": os.path.basename(str(original_location)), "data_id": data_id}
                )
        elif (name, extension) in by_name and by_name[(name, extension)] != str(stored_hash):
            conflicts.append(
                {"name": f"{name}.{extension}" if extension else name, "data_id": data_id}
            )
    if conflicts:
        from cognee.api.v1.exceptions import DocumentUpdateRequiredError

        raise DocumentUpdateRequiredError(conflicts, dataset.id)


__all__ = ["refuse_changed_existing_documents"]
