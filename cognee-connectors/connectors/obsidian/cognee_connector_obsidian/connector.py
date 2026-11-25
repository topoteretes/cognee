"""The Obsidian connector on the shared contract: ``CONNECTOR``.

Obsidian is a local vault: a folder of Markdown notes on the machine that runs the sync, read
by :func:`cognee_connector_obsidian.source.obsidian_source`. There is no account, no token, no
OAuth flow, no webhook and no host adapter in cognee (nothing hosted can reach a vault on a
user's disk, so cognee's integration provider table never listed it). So the descriptor is a
local one (``auth="local"``): the OAuth members of ``DescriptorConnector`` raise
``NotImplementedError``, ``refresh`` returns ``None``, and what works is what a host and the
catalog need: the provider's name, a connection's dataset (``obsidian_<account id>``, from
whatever id a host stores for a vault) and ``sync_scope``, which passes the folders a host
names through as the scope's items with no credential (a vault needs none, so the ``tokens``
a host hands over are empty). It does not enumerate a vault: the source walks it, one
``resource_name`` per vault, so a host builds the source itself
(``obsidian_source(vault_path=..., folders=...)``). Nothing in this module imports cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="obsidian",
    display_name="Obsidian",
    auth="local",
    # No webhooks: a vault is a folder, and the source walks the whole selection on every run.
    # No poll interval either: a sync runs when its caller runs it, as in cognee.
)

# No settings object: the vault path is a parameter of the source factory and nothing is read
# from the environment (settings.py). The default HTTP client is never used: the OAuth members,
# the only ones that would call a provider, refuse before any request, and a vault has no API.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
