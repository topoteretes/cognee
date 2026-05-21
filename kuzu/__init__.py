"""Compatibility shim exposing Ladybug under the legacy Kuzu module name."""

from ladybug import *  # noqa: F403
from ladybug import Connection, __version__  # noqa: F401
from ladybug.database import Database  # noqa: F401
