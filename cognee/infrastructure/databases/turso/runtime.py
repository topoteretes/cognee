"""Driver availability and open errors for the Turso backends. Imports nothing from pyturso."""

from collections.abc import Iterator
from contextlib import contextmanager

INSTALL_HINT = (
    "Turso dependencies are not installed. Install them with "
    "'pip install cognee\"[turso]\"' to use the Turso backends."
)


def require_turso():
    """Import and return the ``turso`` module, or raise a cognee-worded ImportError."""
    try:
        import turso
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return turso


# The engine's message when another process holds the database file's lock
# (matched lowercased): "Locking error: Failed locking file '<path>'. File is
# locked by another process".
_LOCKED_BY_ANOTHER_PROCESS = "file is locked by another process"


def is_locked_by_another_process(error: BaseException) -> bool:
    """True when opening a database failed because another process holds its file lock."""
    return _LOCKED_BY_ANOTHER_PROCESS in str(getattr(error, "orig", None) or error).lower()


@contextmanager
def explain_file_in_use(database_path: str) -> Iterator[None]:
    """Turn the engine's file-lock error into a :class:`TursoDatabaseInUseError`.

    Wrap every place that opens a pyturso connection: the engine locks a file to
    one process, and its own message does not say what to do about it.
    """
    try:
        yield
    except Exception as error:
        if is_locked_by_another_process(error):
            from cognee.infrastructure.databases.exceptions import TursoDatabaseInUseError

            raise TursoDatabaseInUseError(database_path) from error
        raise
