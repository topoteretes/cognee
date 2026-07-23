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

from typing import Optional
from uuid import UUID

from cognee.modules.integrations.credentials import (
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

PROVIDER = "slack"


async def get_by_team(team_id: str) -> Optional[IntegrationCredential]:
    return await get_credential_by_account(PROVIDER, team_id)


async def get_for_user(user_id: UUID) -> Optional[IntegrationCredential]:
    return await get_active_credential_for_user(user_id, PROVIDER)


async def revoke_by_team(team_id: str) -> bool:
    return await revoke_credential_by_account(PROVIDER, team_id)


def is_active(credential: Optional[IntegrationCredential]) -> bool:
    return credential is not None and credential.status == STATUS_ACTIVE


async def resolve_owner_user_id(
    credential: IntegrationCredential, team_id: str, invoking_slack_user_id: str
) -> Optional[UUID]:
    """Resolve which cognee user's memory ``invoking_slack_user_id`` should use.

    Shared by every Slack entry point that touches memory (``/cognee-ask``,
    "Remember this") so the resolution rule lives in one place. Prefers the
    member's own ``/cognee-link`` (real per-person memory); falls back to
    "only the installer" as a stopgap until every member is expected to
    link, which also fails closed for connections made before
    ``installed_by_slack_user_id`` was captured. ``None`` means the invoking
    member isn't authorized to use either.
    """
    member_link = await get_credential_by_account(
        MEMBER_LINK_PROVIDER, member_link_account_id(team_id, invoking_slack_user_id)
    )
    if is_active(member_link):
        return member_link.user_id

    installed_by = (credential.provider_metadata or {}).get("installed_by_slack_user_id")
    if installed_by and installed_by == invoking_slack_user_id:
        return credential.user_id

    return None
