"""Linear connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``linear_source``)
and ``build_linear_service``. This module imports no host: cognee is needed only by
:mod:`cognee_connector_linear.cognee_host`, which is installed with the ``cognee`` extra.
"""

from cognee_connector_linear.connector import CONNECTOR
from cognee_connector_linear.source import build_linear_service, linear_source

__all__ = ["CONNECTOR", "build_linear_service", "linear_source"]
