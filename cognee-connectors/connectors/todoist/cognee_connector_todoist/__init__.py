"""Todoist connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``todoist_source``)
and the client builder a host wraps an API token in (``build_todoist_service``), the two
names ``cognee.tasks.ingestion.connectors`` exported. This module imports no host and no
client library: cognee is needed only by :mod:`cognee_connector_todoist.cognee_host`,
which is installed with the ``cognee`` extra, and ``httpx`` is loaded on the first request.
"""

from cognee_connector_todoist.connector import CONNECTOR
from cognee_connector_todoist.source import build_todoist_service, todoist_source

__all__ = ["CONNECTOR", "build_todoist_service", "todoist_source"]
