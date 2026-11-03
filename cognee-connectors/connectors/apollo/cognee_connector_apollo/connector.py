"""The Apollo connector on the shared contract: ``CONNECTOR``.

Apollo authenticates with an API key the caller hands to
:func:`cognee_connector_apollo.source.apollo_source`: there is no OAuth flow, no webhook
and no host adapter in cognee. So the descriptor is an API-key one (``auth="api_key"``):
the OAuth members of ``DescriptorConnector`` raise ``NotImplementedError``, ``refresh``
returns ``None``, and what works is what a host and the catalog need: the provider's name,
a connection's dataset (``apollo_<team id>``, from the account id a host stores, see
:func:`~.source.apollo_team_id`) and ``sync_scope``, which hands a stored key out as the
sync credential from ``tokens["access_token"]`` for the resources a host names. It does
not enumerate a team: the source lists it, one ``resource_name`` per team, so a host
builds the source itself (``apollo_source(service=build_apollo_service(key), ...)``), the
way cognee's ``modules.integrations.ingestion`` does. Nothing in this module imports
cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="apollo",
    display_name="Apollo",
    auth="api_key",
    # No webhooks: Apollo has no change feed, the source lists every kind in full and
    # fingerprints it. No poll interval either: a sync runs when its caller runs it, as in
    # cognee; a host that schedules syncs picks the cadence against Apollo's per-plan
    # request windows.
)

# No settings object: the key is a parameter of the source factory and is never read from
# the environment (settings.py). The default HTTP client is never used: the OAuth members,
# the only ones that would call the provider, refuse before any request.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
