"""The __DISPLAY_NAME__ connector on the shared contract: ``CONNECTOR``.

A tier-1 descriptor: standard OAuth2 with the account identity readable straight
from the token response. Fill in the provider's endpoints, scopes and the dotted
paths into its token response (docs/PORTING.md, "The contract object"). A provider
whose identity needs one extra API call overrides ``resolve_account`` on a
``DescriptorConnector`` subclass (tier 2); one that is not OAuth2 with a bearer
token subclasses ``Connector`` (tier 3). Nothing in this module imports cognee.
"""

from __MODULE__.settings import __PROVIDER___settings
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="__PROVIDER__",
    display_name="__DISPLAY_NAME__",
    auth="__AUTH__",
    # __OAUTH_FIELDS__
    # A provider with no webhooks is polled; set ``signature_scheme`` and the
    # webhook paths instead when it has them.
    poll_interval_seconds=3600,
)

CONNECTOR = DescriptorConnector(DESCRIPTOR, settings=__PROVIDER___settings)
