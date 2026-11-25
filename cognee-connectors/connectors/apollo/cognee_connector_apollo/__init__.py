"""Apollo.io connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``apollo_source``)
and the service builder a host wraps an API key in (``build_apollo_service``), the two
names ``cognee.tasks.ingestion.connectors`` exported. This module imports no host and no
client library: cognee is needed only by :mod:`cognee_connector_apollo.cognee_host`, which
is installed with the ``cognee`` extra, and ``requests`` is loaded on the first request.
"""

from cognee_connector_apollo.connector import CONNECTOR
from cognee_connector_apollo.source import apollo_source, build_apollo_service

__all__ = ["CONNECTOR", "apollo_source", "build_apollo_service"]
