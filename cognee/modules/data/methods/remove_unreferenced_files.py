from collections.abc import Iterable

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.exceptions import StoredFileNotRemovedError


async def remove_unreferenced_files(locations: Iterable[str | None], outcome: str) -> None:
    """Remove each stored file in ``locations`` that no ``Data`` row points at.

    For the files a document leaves behind when its row moves on, or that were
    saved for a row that never took them. ``remove_data_file_if_unreferenced``
    decides per file: one a row still uses stays, and so does a file outside
    the data root (the user's own).

    A file that cannot be removed raises ``StoredFileNotRemovedError`` and no
    caller catches it. ``outcome`` is the sentence that error opens with: the
    state the document is in at this point, which the caller knows and the
    storage error does not say.
    """
    db_engine = get_relational_engine()
    for location in dict.fromkeys(locations):
        try:
            await db_engine.remove_data_file_if_unreferenced(location)
        except Exception as error:
            raise StoredFileNotRemovedError(
                location, outcome, f"{type(error).__name__}: {error}"
            ) from error
