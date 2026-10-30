"""Asana connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``asana_source``)
and the service builder a host wraps an access token in (``build_asana_service``), the
two names ``cognee.tasks.ingestion.connectors`` exported. This module imports no host and
no client library: cognee is needed only by :mod:`cognee_connector_asana.cognee_host`,
which is installed with the ``cognee`` extra, and ``requests`` is loaded on the first
request.
"""

from cognee_connector_asana.connector import CONNECTOR
from cognee_connector_asana.source import asana_source, build_asana_service

__all__ = ["CONNECTOR", "asana_source", "build_asana_service"]
