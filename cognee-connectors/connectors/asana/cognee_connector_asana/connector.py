"""The Asana connector on the shared contract: ``CONNECTOR``.

Asana authenticates with an access token (a personal access token, or an OAuth access
token a host obtained elsewhere) the caller hands to
:func:`cognee_connector_asana.source.asana_source`: there is no OAuth flow, no webhook and
no host adapter in cognee. So the descriptor is an API-key one (``auth="api_key"``): the
OAuth members of ``DescriptorConnector`` raise ``NotImplementedError``, ``refresh`` returns
``None`` (the source never refreshes a token, a host hands over a fresh one), and what
works is what a host and the catalog need: the provider's name, a connection's dataset
(``asana_<account id>``, from the account id a host stores) and ``sync_scope``, which hands
a stored token out as the sync credential from ``tokens["access_token"]`` with the project
gids a host names as the scope's items. It does not enumerate a workspace: the selection
is ``project_gids``, required, so a host builds the source itself
(``asana_source(service=build_asana_service(token), project_gids=..., ...)``), the way
cognee's ``modules.integrations.ingestion`` does. Nothing in this module imports cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="asana",
    display_name="Asana",
    auth="api_key",
    # No webhooks: the source finds changes through the task listing and the Events API.
    # No poll interval either: a sync runs when its caller runs it, as in cognee; a host
    # that schedules syncs picks the cadence against Asana's per-minute limits.
)

# No settings object: the token is a parameter of the source factory and is never read
# from the environment (settings.py). The default HTTP client is never used: the OAuth
# members, the only ones that would call the provider, refuse before any request.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
