"""Shared plumbing for cognee's Turso (rewrite engine, ``pyturso``) backends.

The relational, graph-as-tables and vector Turso adapters all sit on top of this
package: one SQLAlchemy dialect (``sqlite+cognee_turso://``), one settings class
(``TURSO_*`` env vars), one transaction/retry policy and one file-cleanup rule.
"""

from .config import TursoConfig, get_turso_config
from .dialect import INSTALL_HINT, is_turso_url, register_dialect, require_turso, turso_url
from .files import DATABASE_COMPANION_SUFFIXES, database_file_paths, remove_database_files
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
    "is_turso_url",
    "register_dialect",
    "remove_database_files",
    "require_turso",
    "retry_on_conflict",
    "turso_url",
]
