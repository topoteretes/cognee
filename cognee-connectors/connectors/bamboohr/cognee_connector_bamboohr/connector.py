"""The BambooHR connector on the shared contract: ``CONNECTOR``.

BambooHR authenticates with an API key and a company subdomain, both handed to
:func:`cognee_connector_bamboohr.source.bamboohr_source`: there is no OAuth flow,
no webhook and no host adapter in cognee. So the descriptor is an API-key one
(``auth="api_key"``): the OAuth members of ``DescriptorConnector`` raise
``NotImplementedError``, ``refresh`` returns ``None``, and what works is what a host
and the catalog need: the provider's name, a connection's dataset
(``bamboohr_<account id>``, the account id a host stores is the company's subdomain,
see the source factory's ``subdomain``) and ``sync_scope``, which hands a stored key
out as the sync credential from ``tokens["access_token"]`` for the resources a host
names. It does not enumerate a company: the source lists it, one ``resource_name``
per company, so a host builds the source itself
(``bamboohr_source(service=build_bamboohr_service(key, subdomain), ...)``), the way
cognee's ``modules.integrations.ingestion`` does. Nothing in this module imports
cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="bamboohr",
    display_name="BambooHR",
    auth="api_key",
    # No webhooks: BambooHR has no change feed, the source asks for one report of the
    # whole company and fingerprints it. No poll interval either: a sync runs when
    # its caller runs it, as in cognee; a host that schedules syncs picks the cadence
    # against BambooHR's one-request-per-second pace.
)

# No settings object: the key and the subdomain are parameters of the source factory
# and are never read from the environment (settings.py). The default HTTP client is
# never used: the OAuth members, the only ones that would call the provider, refuse
# before any request.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
