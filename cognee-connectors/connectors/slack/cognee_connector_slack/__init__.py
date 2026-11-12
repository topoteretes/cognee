"""Slack connector for cognee.

Exports the contract object (``CONNECTOR``) and its descriptor. Slack ships no DLT
source: channel history is imported by its own flow, which runs on a cognee server
through :mod:`cognee_connector_slack.cognee_host` (installed with the ``cognee``
extra). This module imports no host.
"""

from cognee_connector_slack.connector import CONNECTOR, DESCRIPTOR

__all__ = ["CONNECTOR", "DESCRIPTOR"]
