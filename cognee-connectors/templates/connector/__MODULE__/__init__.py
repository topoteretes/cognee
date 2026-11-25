"""__DISPLAY_NAME__ connector for cognee.

Exports the contract object (``CONNECTOR``) and the DLT source factory
(``__PROVIDER___source``). This module imports no host: cognee is needed only by
:mod:`__MODULE__.cognee_host`, which is installed with the ``cognee`` extra.
"""

from __MODULE__.connector import CONNECTOR
from __MODULE__.source import __PROVIDER___source

__all__ = ["CONNECTOR", "__PROVIDER___source"]
