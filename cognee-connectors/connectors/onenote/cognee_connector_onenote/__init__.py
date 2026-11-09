"""OneNote connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``onenote_source``) and
the notebook listing a host offers a picker from (``list_onenote_notebooks``), the two names
``cognee.tasks.ingestion.connectors`` exported. This module imports no host and no client
library: cognee is needed only by :mod:`cognee_connector_onenote.cognee_host`, which is
installed with the ``cognee`` extra, ``httpx`` is loaded on the first Graph request and
``beautifulsoup4`` when the first page is rendered.
"""

from cognee_connector_onenote.connector import CONNECTOR
from cognee_connector_onenote.source import list_onenote_notebooks, onenote_source

__all__ = ["CONNECTOR", "list_onenote_notebooks", "onenote_source"]
