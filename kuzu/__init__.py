"""Compatibility shim exposing Ladybug under the legacy Kuzu module name."""

# Registers the Windows DLL search path ladybug's native extension needs; must
# precede the ``ladybug`` imports below. See cognee_db_workers/_windows_openssl.py.
import cognee_db_workers

from ladybug import *
from ladybug import Connection, __version__
from ladybug.database import Database
