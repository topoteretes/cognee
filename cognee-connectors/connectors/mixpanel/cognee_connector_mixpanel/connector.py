"""The Mixpanel connector on the shared contract: ``CONNECTOR``.

Mixpanel authenticates with a service account or a legacy project API secret that the
caller hands to :func:`cognee_connector_mixpanel.source.mixpanel_source`: there is no
OAuth flow, no webhook and no host adapter in cognee. So the descriptor is an API-key
one (``auth="api_key"``): the OAuth members of ``DescriptorConnector`` raise
``NotImplementedError``, ``refresh`` returns ``None``, and what works is what a host
and the catalog need: the provider's name, a connection's dataset
(``mixpanel_<project id>``, from the account id a host stores, see
:func:`~.source.mixpanel_project_id`) and ``sync_scope``, which hands a stored secret
out as the sync credential from ``tokens["access_token"]`` for the resources a host
names. It does not enumerate a project: the source resolves it (one request), one
``resource_name`` per project, so a host builds the source itself
(``mixpanel_source(service=build_mixpanel_service(...), ...)``), the way cognee's
``modules.integrations.ingestion`` does. Nothing in this module imports cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="mixpanel",
    display_name="Mixpanel",
    auth="api_key",
    # No webhooks: Mixpanel has no change feed for definitions, the source lists every
    # kind in full and fingerprints it. No poll interval either: a sync runs when its
    # caller runs it, as in cognee; a host that schedules syncs picks the cadence
    # against the Query API's 60-queries-an-hour limit.
)

# No settings object: the credential is a parameter of the source factory and is never
# read from the environment (settings.py). The default HTTP client is never used: the
# OAuth members, the only ones that would call the provider, refuse before any request.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
