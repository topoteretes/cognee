"""GitHub connector for cognee.

Exports the contract object (``CONNECTOR``), a Tier 3 connector for GitHub App
installations. There is no DLT source: a GitHub sync is cognee's code-graph
ingestion (``remember(content_type="code")``) over the installation's
repositories, which a host drives from the ``SyncScope`` the connector builds.
This module imports no host: cognee is needed only by
:mod:`cognee_connector_github.cognee_host`, which is installed with the
``cognee`` extra.
"""

from cognee_connector_github.connector import CONNECTOR, GithubConnector

__all__ = ["CONNECTOR", "GithubConnector"]
