"""BambooHR connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``bamboohr_source``)
and the service builder a host wraps an API key and a company subdomain in
(``build_bamboohr_service``), the names ``cognee.tasks.ingestion.connectors``
exported. This module imports no host and no client library: cognee is needed only by
:mod:`cognee_connector_bamboohr.cognee_host`, which is installed with the ``cognee``
extra, and ``requests`` is loaded on the first request.
"""

from cognee_connector_bamboohr.connector import CONNECTOR
from cognee_connector_bamboohr.source import bamboohr_source, build_bamboohr_service

__all__ = ["CONNECTOR", "bamboohr_source", "build_bamboohr_service"]
