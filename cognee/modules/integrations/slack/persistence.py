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

PROVIDER = "slack"


async def get_by_team(team_id: str) -> Optional[IntegrationCredential]:
    return await get_credential_by_account(PROVIDER, team_id)


async def get_for_user(user_id: UUID) -> Optional[IntegrationCredential]:
    return await get_active_credential_for_user(user_id, PROVIDER)


async def revoke_by_team(team_id: str) -> bool:
    return await revoke_credential_by_account(PROVIDER, team_id)


def is_active(credential: Optional[IntegrationCredential]) -> bool:
    return credential is not None and credential.status == STATUS_ACTIVE
