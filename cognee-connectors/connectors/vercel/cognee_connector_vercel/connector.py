"""Vercel uses a caller-supplied bearer token and explicitly selected projects.

There is no OAuth flow, resource picker or automatic polling. Hosts construct one
source per project and pass team_id/account_id themselves; the generic sync scope
only forwards the selected project IDs and token.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(provider="vercel", display_name="Vercel", auth="api_key")
CONNECTOR = DescriptorConnector(DESCRIPTOR)
