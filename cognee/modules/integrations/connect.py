"""Generic OAuth installation completion, shared by every provider.

Exchange code -> parse response -> persist credential is the same three-step
dance regardless of provider; the only per-provider pieces are
``exchange_code`` and ``parse_installation`` on the
:class:`~cognee.modules.integrations.base.OAuthIntegration` itself. Writing
that orchestration once here means a new adapter never has to repeat it —
see :mod:`cognee.api.v1.integrations.routers.get_integrations_router`, whose
callback endpoint calls this for whichever provider is in the URL.
"""

from uuid import UUID

from cognee.modules.integrations.base import OAuthIntegration
from cognee.modules.integrations.credentials import upsert_credential
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


async def complete_installation(
    integration: OAuthIntegration, *, code: str, user_id: UUID
) -> IntegrationCredential:
    """Exchange ``code`` for tokens and persist the resulting credential.

    Raises whatever ``exchange_code``/``parse_installation`` raise (a
    provider-specific ``RuntimeError``/``ValueError`` for a rejected or
    malformed exchange) or
    :class:`~cognee.modules.integrations.credentials.CrossUserConflictError`
    from the upsert — callers (the callback endpoint) translate both into a
    redirect outcome, never a raw 500.
    """
    token_response = await integration.exchange_code(code)
    installation = integration.parse_installation(token_response)

    return await upsert_credential(
        provider=integration.provider,
        user_id=user_id,
        provider_account_id=installation.provider_account_id,
        token_payload=installation.token_payload,
        account_label=installation.account_label,
        auth_type=installation.auth_type,
        scopes=installation.scopes,
        provider_metadata=installation.provider_metadata,
        token_expires_at=installation.token_expires_at,
    )
