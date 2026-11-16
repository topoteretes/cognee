"""The Todoist connector on the shared contract: ``CONNECTOR``.

Todoist authenticates with an API token (from Settings → Integrations in the Todoist
app) the caller hands to :func:`cognee_connector_todoist.source.todoist_source`: there is
no OAuth flow, no webhook and no host adapter in cognee. So the descriptor is an API-key
one (``auth="api_key"``): the OAuth members of ``DescriptorConnector`` raise
``NotImplementedError``, ``refresh`` returns ``None`` (the source never refreshes a
token, a host hands over a fresh one), and what works is what a host and the catalog
need: the provider's name and a connection's dataset (``todoist_<account id>``, from the
account id a host stores).

The one member Todoist overrides is ``sync_scope``: a plain descriptor refuses a
selection-less sync because it cannot enumerate an account without a provider call.
Todoist can: the Sync API's ``"*"`` full sync returns the whole account, and the source
filters the selection in flight. So ``resources=None`` (the connect path) is a valid
whole-account sync whose scope collapses to empty ``items`` on purpose, and a list of
project ids limits it. Nothing in this module imports cognee.
"""

from cognee_connectors_core.base import SyncScope
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector


class TodoistConnector(DescriptorConnector):
    """The API-key descriptor with the one override Todoist needs: full-account sync.

    The base refuses ``resources=None`` because a descriptor cannot enumerate an
    account. Todoist needs no enumeration call: a sync with the ``"*"`` cursor returns
    the entire account, and ``todoist_source`` applies the selection while it renders.
    """

    async def sync_scope(self, credential, tokens, *, resources=None) -> SyncScope:
        if resources is not None and not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )
        return SyncScope(
            dataset_name=self.dataset_name(credential),
            content_type=self.descriptor.content_type,
            items=list(resources or []),
            credential=self._access_token(credential, tokens),
        )


DESCRIPTOR = ConnectorDescriptor(
    provider="todoist",
    display_name="Todoist",
    auth="api_key",
    # No webhooks: the Sync API's sync_token cursor finds changes, including deletions.
    # No poll interval either: a sync runs when its caller runs it, as in cognee; a host
    # that schedules syncs picks the cadence against Todoist's rate limits.
)

# No settings object: the token is a parameter of the source factory and is never read
# from the environment (settings.py). The default HTTP client is never used: the OAuth
# members, the only ones that would call the provider, refuse before any request.
CONNECTOR = TodoistConnector(DESCRIPTOR)
