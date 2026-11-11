"""The Slack connector on the shared contract: ``CONNECTOR``.

Derived from cognee's Slack adapter and OAuth module
(cognee.modules.integrations.slack.adapter and .oauth, cognee dev, 2026-10-27): the
same ``oauth/v2/authorize`` and ``oauth.v2.access`` endpoints, the same bot scopes,
and the fields the adapter reads out of a token response — ``team.id`` as the
account webhooks route by, ``team.name`` as its label, the granted ``scope`` string,
and ``bot_user_id`` / ``enterprise.id`` / ``authed_user.id`` kept in the clear as
provider metadata. Inbound requests are verified with the ``slack_v0`` scheme over
``X-Slack-Signature`` / ``X-Slack-Request-Timestamp`` keyed with
``SLACK_SIGNING_SECRET`` (:mod:`cognee_connector_slack.webhooks`).

A tier-1 descriptor: nothing here imports cognee, and the hand-written Slack app
(slash commands, "Remember this", App Home, the history import) stays in
:mod:`cognee_connector_slack.cognee_host`. Two things that adapter does are outside
what a descriptor can say. An Enterprise Grid org-level install, whose token
response carries ``enterprise.id`` and no ``team``, is identified by the adapter's
fallback, not by ``account_id_path``. And Slack names an event's type inside the
body (``event.type``), not in a header, so ``app_uninstalled`` / ``tokens_revoked``
are handled by the host's Events API handler rather than ``webhook_revoke_events``.
"""

from cognee_connector_slack.settings import slack_settings
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.signatures import SignatureScheme

#: Slack's OAuth v2 endpoints (https://docs.slack.dev/authentication/installing-with-oauth).
AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
ACCESS_URL = "https://slack.com/api/oauth.v2.access"

# History is fetched only for explicitly selected channels. Metadata scopes
# also let imports verify both bot and connecting-user membership. Existing
# installations must re-authorize to grant these additional scopes.
BOT_SCOPES = (
    "commands,chat:write,im:write,channels:read,channels:history,groups:read,groups:history"
)

DESCRIPTOR = ConnectorDescriptor(
    provider="slack",
    display_name="Slack",
    authorize_url=AUTHORIZE_URL,
    token_url=ACCESS_URL,
    scopes=BOT_SCOPES,
    # oauth.v2.access answers for the workspace the app was installed into.
    account_id_path="team.id",
    account_label_path="team.name",
    scopes_path="scope",
    # Non-secret, Slack-specific fields the host needs later (posting messages,
    # Grid-awareness, the installer's identity for authorization checks).
    metadata_paths=(
        ("bot_user_id", "bot_user_id"),
        ("enterprise_id", "enterprise.id"),
        ("installed_by_slack_user_id", "authed_user.id"),
    ),
    # Slack returns HTTP 200 with ``ok: false`` and an ``error`` code on a
    # rejected exchange, so the status alone cannot be trusted.
    ok_path="ok",
    error_path="error",
    signature_scheme=SignatureScheme.SLACK_V0,
    signature_header="x-slack-signature",
    signature_timestamp_header="x-slack-request-timestamp",
    # An Events API envelope names the workspace and carries a per-delivery id;
    # Slack retries a delivery up to three times, so ``event_id`` is what to
    # dedupe on. Slash commands and interactive payloads are form-encoded and
    # go through the host's own routes, never through ``handle_webhook``.
    webhook_account_id_path="team_id",
    webhook_delivery_id_path="event_id",
)

CONNECTOR = DescriptorConnector(DESCRIPTOR, settings=slack_settings)
