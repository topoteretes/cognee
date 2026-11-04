"""Obsidian connector for cognee.

Exports the contract object (``CONNECTOR``) and the DLT source factory (``obsidian_source``),
the name ``cognee.tasks.ingestion.connectors`` exported. This module imports no host and no
library beyond the contract: cognee is needed only by
:mod:`cognee_connector_obsidian.cognee_host`, which is installed with the ``cognee`` extra, and
``dlt`` and PyYAML are loaded when a source is built and a note is read.
"""

from cognee_connector_obsidian.connector import CONNECTOR
from cognee_connector_obsidian.source import obsidian_source

__all__ = ["CONNECTOR", "obsidian_source"]
