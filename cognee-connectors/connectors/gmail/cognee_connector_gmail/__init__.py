"""Gmail connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``gmail_source``)
and the client builders cognee's connector package exported. This module imports no host:
cognee is needed only by :mod:`cognee_connector_gmail.cognee_host`, which is installed with
the ``cognee`` extra and which nothing here imports.
"""

from cognee_connector_gmail.connector import CONNECTOR, DESCRIPTOR, GmailConnector
from cognee_connector_gmail.source import (
    build_gmail_service,
    build_gmail_service_from_access_token,
    gmail_source,
)

__all__ = [
    "CONNECTOR",
    "DESCRIPTOR",
    "GmailConnector",
    "build_gmail_service",
    "build_gmail_service_from_access_token",
    "gmail_source",
]
