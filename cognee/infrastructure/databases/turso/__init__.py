"""Shared plumbing for cognee's Turso (rewrite engine, ``pyturso``) backends.

The relational, graph-as-tables and vector Turso adapters all sit on top of this
package: one SQLAlchemy dialect (``sqlite+cognee_turso://``), one settings class
(``TURSO_*`` env vars), one transaction/retry policy and one file-cleanup rule.
"""

from .config import TursoConfig, get_turso_config
from .runtime import INSTALL_HINT, require_turso
from .files import (
    DATABASE_COMPANION_SUFFIXES,
    database_file_paths,
    remove_database_files,
    remove_database_files_from_storage,
)
from .transactions import (
    apply_pragmas,
    begin_statement,
    configure_engine,
    connect_args_for_mode,
    connect_pragmas,
    exclusive_transaction,
    install_connect_pragmas,
    install_transaction_hook,
    is_retryable_conflict,
    retry_on_conflict,
)

DIALECT_NAME = "sqlite"
DRIVER_NAME = "cognee_turso"
_URL_PREFIX = f"{DIALECT_NAME}+{DRIVER_NAME}://"


# The dialect module imports pyturso at load time. This package is reached from
# cognee's core import path (the dataset-handler registry imports the Turso
# handlers), so the dialect is loaded lazily: installations without the turso
# extra must import cognee, and only fail when a Turso engine is actually built.


def register_dialect() -> None:
    """Make ``sqlite+cognee_turso://`` resolvable by ``create_async_engine``. Idempotent."""
    from .dialect import register_dialect as _register

    _register()


def turso_url(database_path: str) -> str:
    """Return the SQLAlchemy URL for a local Turso database file (or ``:memory:``).

    Registers the dialect as a side effect so callers can hand the URL straight to
    ``create_async_engine``. An absolute path yields ``sqlite+cognee_turso:////abs``,
    the same four-slash shape the SQLite branch produces.
    """
    register_dialect()
    return f"{_URL_PREFIX}/{database_path}"


__all__ = [
    "DATABASE_COMPANION_SUFFIXES",
    "INSTALL_HINT",
    "TursoConfig",
    "apply_pragmas",
    "begin_statement",
    "configure_engine",
    "connect_args_for_mode",
    "connect_pragmas",
    "database_file_paths",
    "exclusive_transaction",
    "get_turso_config",
    "install_connect_pragmas",
    "install_transaction_hook",
    "is_retryable_conflict",
    "register_dialect",
    "remove_database_files",
    "remove_database_files_from_storage",
    "require_turso",
    "retry_on_conflict",
    "turso_url",
]
