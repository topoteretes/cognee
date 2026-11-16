"""arXiv connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``arxiv_source``)
and the service builder (``build_arxiv_service``). This module imports no host and no
client library: cognee is needed only by :mod:`cognee_connector_arxiv.cognee_host`,
which is installed with the ``cognee`` extra, and ``requests`` is loaded on the first
request.
"""

from cognee_connector_arxiv.connector import CONNECTOR
from cognee_connector_arxiv.source import arxiv_source, build_arxiv_service

__all__ = ["CONNECTOR", "arxiv_source", "build_arxiv_service"]
