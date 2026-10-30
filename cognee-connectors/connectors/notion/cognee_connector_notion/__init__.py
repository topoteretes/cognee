"""Notion connector for cognee.

Exports the contract object (``CONNECTOR``) and the DLT source factory
(``notion_source``). This module imports no host: cognee is needed only by
:mod:`cognee_connector_notion.cognee_host`, which is installed with the ``cognee``
extra and which nothing here imports.
"""

from cognee_connector_notion.connector import CONNECTOR
from cognee_connector_notion.source import notion_source

__all__ = ["CONNECTOR", "notion_source"]
