"""Slack adapter over the generic credential store.

Exposes team-id-keyed lookups the webhook handlers route through. Persisting
a completed OAuth exchange now goes through the generic
:func:`cognee.modules.integrations.connect.complete_installation` +
:class:`~cognee.modules.integrations.slack.adapter.SlackIntegration` instead
of a Slack-specific ``save_installation`` — the field extraction that used
to live here (splitting Slack's ``oauth.v2.access`` response into token
payload / metadata / account id) has moved onto that adapter, which is the
one place a provider's response-shape knowledge should live.
"""

from uuid import UUID

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.credentials import (
    REVOKED_BY_PROVIDER,
    STATUS_ACTIVE,
    get_active_credential_for_user,
    get_credential_by_account,
    revoke_credential_by_account,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.slack.handle_slack_link import (
    MEMBER_LINK_PROVIDER,
    member_link_account_id,
)
from cognee.modules.users.methods import get_user

PROVIDER = "slack"


async def get_by_team(team_id: str) -> IntegrationCredential | None:
    return await get_credential_by_account(PROVIDER, team_id)


async def get_for_user(user_id: UUID) -> IntegrationCredential | None:
    return await get_active_credential_for_user(user_id, PROVIDER)


async def revoke_by_team(team_id: str) -> bool:
    return await revoke_credential_by_account(PROVIDER, team_id, revoked_by=REVOKED_BY_PROVIDER)


def is_active(credential: IntegrationCredential | None) -> bool:
    return credential is not None and credential.status == STATUS_ACTIVE


CHANNEL_NOT_ENABLED_MESSAGE = (
    "Not enabled in this channel\nAsk the person who connected Slack to allow Cognee here."
)


def channel_allowed(credential: IntegrationCredential, channel_id: str) -> bool:
    """Whether the workspace's channel allowlist lets Cognee act in ``channel_id``.

    An empty allowlist means unrestricted: channel scoping is opt-in via
    ``PUT /api/v1/slack/channels``. Every entry point that reads or writes memory on
    a member's request (``@cognee``, "Remember this") asks this one function; a
    missing channel id fails closed once an allowlist is set.
    """
    allowed = (credential.provider_metadata or {}).get("allowed_channel_ids") or []
    return not allowed or channel_id in allowed


async def resolve_owner_user_id(
    credential: IntegrationCredential, team_id: str, invoking_slack_user_id: str
) -> UUID | None:
    """Resolve which cognee user's memory ``invoking_slack_user_id`` should use.

    Shared by every Slack entry point that touches memory (``@cognee``,
    "Remember this") so the resolution rule lives in one place. Prefers the
    member's own link (Sign in with Slack, real per-person memory); falls back to
    "only the installer" as a stopgap until every member is expected to
    link, which also fails closed for connections made before
    ``installed_by_slack_user_id`` was captured. ``None`` means the invoking
    member isn't authorized to use either.
    """
    member_link = await get_credential_by_account(
        MEMBER_LINK_PROVIDER, member_link_account_id(team_id, invoking_slack_user_id)
    )
    if is_active(member_link):
        return await _if_account_active(member_link.user_id)

    installed_by = (credential.provider_metadata or {}).get("installed_by_slack_user_id")
    if installed_by and installed_by == invoking_slack_user_id:
        return await _if_account_active(credential.user_id)

    return None


async def _if_account_active(user_id: UUID) -> UUID | None:
    """A deactivated cognee account is refused here as every API route refuses it (G53),
    and a deleted one the same way (G63): raising would answer Slack with a 500, which it
    sends again."""
    try:
        user = await get_user(user_id)
    except EntityNotFoundError:
        return None
    return user_id if user.is_active else None
