from collections.abc import Iterable

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.exceptions import StoredFileNotRemovedError
from cognee.shared.logging_utils import get_logger

logger = get_logger("remove_unreferenced_files")


async def remove_unreferenced_files(locations: Iterable[str | None], outcome: str) -> None:
    """Remove each stored file in ``locations`` that no ``Data`` row points at.

    For the files a document leaves behind when its row moves on, or that were
    staged for a row that never took them. A file a row still uses stays, and
    so does a file outside the data root (the user's own). The references of
    all the files are read in one query.

    Every file is tried. The ones that could not be removed are then raised
    together as one ``StoredFileNotRemovedError``, each with its own error, and
    no caller catches it: stopping at the first failure would leave the rest
    on disk with nobody told, and nothing finds them later. ``outcome`` is the
    sentence that error opens with: the state the document is in at this
    point, which the caller knows and a storage error does not say.
    """
    db_engine = get_relational_engine()
    candidates = [
        location
        for location in dict.fromkeys(locations)
        if location and db_engine.is_managed_data_file(location)
    ]
    if not candidates:
        return

    try:
        referenced = await db_engine.referenced_data_files(candidates)
    except Exception as error:
        # No file could be checked, so none was removed: all of them are reported.
        reason = f"{type(error).__name__}: {error}"
        raise StoredFileNotRemovedError(dict.fromkeys(candidates, reason), outcome) from error

    not_removed: dict[str, str] = {}
    first_error: Exception | None = None
    for location in candidates:
        if location in referenced:
            continue
        try:
            await db_engine.remove_data_file(location)
        except Exception as error:
            # Kept, not dropped: the loop goes on to the other files, and every
            # failure is raised together below.
            logger.exception("Could not remove stored file %s", location)
            not_removed[location] = f"{type(error).__name__}: {error}"
            first_error = first_error or error
    if not_removed:
        raise StoredFileNotRemovedError(not_removed, outcome) from first_error
