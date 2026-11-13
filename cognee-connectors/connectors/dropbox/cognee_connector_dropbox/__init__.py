"""Dropbox connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``dropbox_source``) and
the two helpers ``cognee.tasks.ingestion.connectors`` exports next to it: the service builder a
host wraps an access token in (``build_dropbox_service``) and the account lookup
(``dropbox_account_id``). This module imports no host: cognee's own Dropbox integration (OAuth
routes, credential store, the sync runner) stays in cognee, so the package has no
``cognee_host`` subpackage. The source's ``httpx`` client is loaded on the first request.
"""

from cognee_connector_dropbox.connector import CONNECTOR
from cognee_connector_dropbox.source import (
    build_dropbox_service,
    dropbox_account_id,
    dropbox_source,
)

__all__ = ["CONNECTOR", "build_dropbox_service", "dropbox_account_id", "dropbox_source"]
