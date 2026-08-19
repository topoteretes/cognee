"""Compatibility shim exposing Ladybug under the legacy Kuzu module name."""

# Registers the Windows DLL search path ladybug's native extension needs; must
# precede the ``ladybug`` imports below. See cognee_db_workers/_windows_openssl.py.
import cognee_db_workers  # noqa: F401

from ladybug import *  # noqa: F403
from ladybug import Connection, __version__  # noqa: F401
from ladybug.database import Database  # noqa: F401
