import asyncio
import sys
from collections.abc import Iterable

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.exceptions import StoredFileNotRemovedError
from cognee.shared.logging_utils import get_logger

logger = get_logger("remove_unreferenced_files")

# How many removals run at once. Their storage calls run on the process's shared
# thread pool, and local storage does them on the event loop itself, so the
# files of a large re-sync started all together make every other request in the
# worker wait behind them. A width, not a cutoff: every file is still removed.
_REMOVALS_AT_ONCE = 16


async def remove_unreferenced_files(locations: Iterable[str | None], outcome: str) -> None:
    """Remove each stored file in ``locations`` that no ``Data`` row points at.

    For the files a document leaves behind when its row moves on, or that were
    staged for a row that never took them. A file a row still uses stays, and
    so does a file outside the data root (the user's own). The references of
    all the files are read in one query, and the removals run
    ``_REMOVALS_AT_ONCE`` at a time: each is a few storage round trips, and a
    re-sync can release two files per changed document while it holds the
    dataset lock.

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

    unreferenced = [location for location in candidates if location not in referenced]
    errors: list[Exception | None] = []
    # One slice after another, not a sliding window: awaiting each slice is what
    # gives the event loop its turn when the removals themselves never wait
    # (local storage).
    for start in range(0, len(unreferenced), _REMOVALS_AT_ONCE):
        errors += await asyncio.gather(
            *(
                _remove_data_file(db_engine, location)
                for location in unreferenced[start : start + _REMOVALS_AT_ONCE]
            )
        )
    not_removed = {
        location: f"{type(error).__name__}: {error}"
        for location, error in zip(unreferenced, errors)
        if error is not None
    }
    if not_removed:
        first_error = next(error for error in errors if error is not None)
        # A removal runs in its own task, which does not see the error this call
        # may be cleaning up after (a failed write, a refusal). Raised in this
        # frame it would carry that error as its context; link it, so the
        # traceback still shows why the document is in the state ``outcome`` names.
        if first_error.__context__ is None:
            first_error.__context__ = sys.exc_info()[1]
        raise StoredFileNotRemovedError(not_removed, outcome) from first_error


async def _remove_data_file(db_engine: SQLAlchemyAdapter, location: str) -> Exception | None:
    """Remove one stored file and return the error that kept it, if any.

    Returned, not raised: the other removals go on, and every failure is
    raised together by the caller.
    """
    try:
        await db_engine.remove_data_file(location)
    except Exception as error:
        logger.exception("Could not remove stored file %s", location)
        return error
    return None
