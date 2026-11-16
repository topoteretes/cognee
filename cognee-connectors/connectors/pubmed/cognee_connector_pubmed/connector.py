"""The PubMed connector on the shared contract: ``CONNECTOR``.

PubMed authenticates with an optional NCBI API key the caller hands to
:func:`cognee_connector_pubmed.source.pubmed_source`: there is no OAuth flow, no
webhook and no host adapter in cognee. So the descriptor is an API-key one
(``auth="api_key"``): the OAuth members of ``DescriptorConnector`` raise
``NotImplementedError``, ``refresh`` returns ``None``, and what works is what a host
and the catalog need: the provider's name, a connection's dataset (``pubmed_<query
slug>``, from the account id a host stores) and ``sync_scope``, which hands a stored
key out as the sync credential from ``tokens["access_token"]`` for the resources a
host names. It does not enumerate a query: the source lists it, one
``resource_name`` per query, so a host builds the source itself
(``pubmed_source(service=build_pubmed_service(key), ...)``), the way cognee's
``modules.integrations.ingestion`` does. Nothing in this module imports cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="pubmed",
    display_name="PubMed",
    auth="api_key",
    # No webhooks and no polling: the source lists its window when its caller runs
    # it, as in cognee; a host that schedules syncs picks the cadence against
    # NCBI's request limits (3/s without a key, 10/s with one).
)

# No settings object: the key is a parameter of the source factory and is never read
# from the environment (settings.py). The default HTTP client is never used: the
# OAuth members, the only ones that would call the provider, refuse before any
# request.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
