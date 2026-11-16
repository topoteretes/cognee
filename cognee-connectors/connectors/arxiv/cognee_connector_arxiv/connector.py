"""The arXiv connector on the shared contract: ``CONNECTOR``.

arXiv is a public API: no account, no token, no OAuth flow, no webhook and no host
adapter in cognee, so the descriptor is a local one (``auth="local"``), the shape the
contract gives providers that need no credential: the OAuth members of
:class:`~cognee_connectors_core.oauth2.DescriptorConnector` raise
``NotImplementedError``, ``refresh`` returns ``None``, and what works is what a host
and the catalog need: the provider's name, a connection's dataset (``arxiv_<account
id>``, from whatever id a host stores for a selection) and ``sync_scope``, which
hands the categories or authors a host names to the source with no credential (the
API takes none, so the ``tokens`` a host hands over are empty). It does not enumerate
papers: the source pages the API, one ``resource_name`` per selection, so a host
builds the source itself (``arxiv_source(categories=...)``), the way cognee's
``modules.integrations.ingestion`` does. Nothing in this module imports cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="arxiv",
    display_name="arXiv",
    auth="local",
    # No webhooks: arXiv has no change feed, the source pages a submittedDate window.
    # No poll interval either: a sync runs when its caller runs it, as in cognee.
)

# No settings object: the selection and the rate limit are parameters of
# :func:`cognee_connector_arxiv.source.arxiv_source` and nothing is read from the
# environment (settings.py). The default HTTP client is never used: the OAuth members,
# the only ones that would call the provider, refuse before any request, and the
# source's client is built by the source factory.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
