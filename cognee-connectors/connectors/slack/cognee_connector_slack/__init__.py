"""Slack connector for cognee.

Exports the contract object (``CONNECTOR``), its descriptor and the DLT source factory
(``slack_source``). This module imports no host: cognee is needed only by
:mod:`cognee_connector_slack.cognee_host` (the Slack app and the channel sync on a
cognee server), which is installed with the ``cognee`` extra and which nothing here
imports.
"""

from cognee_connector_slack.connector import CONNECTOR, DESCRIPTOR
from cognee_connector_slack.source import slack_source

__all__ = ["CONNECTOR", "DESCRIPTOR", "slack_source"]
