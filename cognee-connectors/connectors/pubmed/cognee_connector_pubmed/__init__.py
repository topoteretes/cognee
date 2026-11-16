"""PubMed connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``pubmed_source``)
and the service builder a host wraps an API key in (``build_pubmed_service``), the two
names ``cognee.tasks.ingestion.connectors`` exported. This module imports no host and no
client library: cognee is needed only by :mod:`cognee_connector_pubmed.cognee_host`,
which is installed with the ``cognee`` extra, and ``requests`` is loaded on the first
request.
"""

from cognee_connector_pubmed.connector import CONNECTOR
from cognee_connector_pubmed.source import build_pubmed_service, pubmed_source

__all__ = ["CONNECTOR", "build_pubmed_service", "pubmed_source"]
