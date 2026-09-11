"""Refuse an add() of a file the dataset already holds with other content.

add() creates documents and leaves existing ones alone. A file that matches a
stored document by origin but not by content is an update, and updates go
through update() so the document keeps its id and its graph is replaced in
place instead of a second copy being minted. Identical content is not an
error: ingestion deduplicates it by content hash and the re-add is a no-op.

This runs once, over the whole request, before the add pipeline starts: the
pipeline processes items one at a time, so a check inside it would refuse the
offending file only after the files before it were written.

Origins (a local file's path, an upload's filename) and the lookup are shared
with update(), which infers its target the same way — see
``cognee.modules.ingestion.identify_by_origin``. Raw text never matches: it is
named by its content hash, so a changed text is a new document by
construction. Pinned items (DLT manifests, code-repo manifests, update()'s own
re-add) name their document and are exempt.
"""

from typing import Any

from cognee.modules.ingestion.identify_by_origin import (
    Origin,
    expand_directories,
    find_by_origin,
    origin_of,
)
from cognee.modules.users.models import User
from cognee.tasks.ingestion.data_item import DataItem


async def _unpinned_origins(data: Any) -> list[Origin]:
    """Origins of the inputs that do not name their document already."""
    items = data if isinstance(data, list) else [data]
    unpinned = []
    for item in items:
        if isinstance(item, DataItem):
            if item.data_id is not None:
                continue
            item = item.data
        unpinned.append(item)

    origins: list[Origin] = []
    for item in expand_directories(unpinned):
        origin = await origin_of(item)
        if origin is not None:
            origins.append(origin)
    return origins


async def refuse_changed_existing_documents(data: Any, user: User, dataset) -> None:
    """Raise ``DocumentUpdateRequiredError`` for every input that is an update in disguise."""
    origins = await _unpinned_origins(data)
    matches = await find_by_origin(origins, user, dataset.id)

    conflicts = [
        {"name": origin.label, "data_id": match.data_id}
        for origin in origins
        for match in matches[origin]
        if match.content_hash != origin.content_hash
    ]
    if conflicts:
        from cognee.api.v1.exceptions import DocumentUpdateRequiredError

        raise DocumentUpdateRequiredError(conflicts, dataset.id)


__all__ = ["refuse_changed_existing_documents"]
