"""Mixpanel connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory (``mixpanel_source``)
and the service builder a host wraps a credential in (``build_mixpanel_service``). This
module imports no host and no client library: cognee is needed only by
:mod:`cognee_connector_mixpanel.cognee_host`, which is installed with the ``cognee``
extra, and ``requests`` is loaded on the first request.
"""

from cognee_connector_mixpanel.connector import CONNECTOR
from cognee_connector_mixpanel.source import build_mixpanel_service, mixpanel_source

__all__ = ["CONNECTOR", "build_mixpanel_service", "mixpanel_source"]
