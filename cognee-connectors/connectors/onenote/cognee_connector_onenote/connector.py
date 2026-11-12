"""The OneNote connector on the shared contract: ``CONNECTOR``.

OneNote is read through Microsoft Graph with a delegated access token the caller hands to
:func:`cognee_connector_onenote.source.onenote_source` (a string, or a zero-argument callable
returning one): the source never logs in and never refreshes a token, and cognee ships no OAuth
adapter, no webhook and no settings for it (its provider table names the source alone, with no
service builder). So the descriptor is an API-key one (``auth="api_key"``): the OAuth members
of ``DescriptorConnector`` raise ``NotImplementedError``, ``refresh`` returns ``None`` (a host
that holds a Microsoft refresh token hands a fresh access token over itself), and what works is
what a host and the catalog need: the provider's name, a connection's dataset
(``onenote_<account id>``, from the stable account id a host stores, the same ``account_id``
the source takes) and ``sync_scope``, which hands a stored token out as the sync credential
from ``tokens["access_token"]`` with the notebook ids a host names as the scope's items. It
does not enumerate an account: :func:`~.source.list_onenote_notebooks` does, and one source is
one notebook, so a host builds each source itself
(``onenote_source(token, notebook_id=..., account_id=..., resource_name=...)``), the way
cognee's ``modules.integrations.ingestion`` does. Nothing in this module imports cognee.
"""

from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector

DESCRIPTOR = ConnectorDescriptor(
    provider="onenote",
    display_name="OneNote",
    auth="api_key",
    # No webhooks: Graph has no delta feed for OneNote, the source lists the whole notebook
    # and fingerprints every page. No poll interval either: a sync runs when its caller runs
    # it, as in cognee; a host that schedules syncs picks the cadence against Graph's
    # per-user OneNote throttling (120 requests a minute, 400 an hour).
)

# No settings object: the token is a parameter of the source factory and is never read from
# the environment (settings.py). The default HTTP client is never used: the OAuth members, the
# only ones that would call the provider, refuse before any request.
CONNECTOR = DescriptorConnector(DESCRIPTOR)
