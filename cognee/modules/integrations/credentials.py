"""Generic persistence for third-party connector credentials.

Provider-agnostic CRUD over the single ``integration_credentials`` table.
Connector-specific code (e.g. the Slack layer) maps its OAuth response into
the ``token_payload`` + metadata these functions accept, then routes inbound
webhooks back via :func:`get_credential_by_account`. Token material is
encrypted here and never returned in plaintext except by
:func:`decrypt_token_payload`, which callers invoke only at the moment they
need to call the provider's API.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from sqlalchemy import select

from cognee.modules.integrations.crypto import decrypt_credentials, encrypt_credentials
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_REVOKED = "revoked"


class CrossUserConflictError(Exception):
    """A different user already holds an active connection for this external account.

    A workspace maps to exactly one user (UNIQUE(provider,
    provider_account_id)). Silently reassigning it on reconnect would let one
    user take over another's connection with no trace — so we refuse and let
    the original owner disconnect first.
    """


async def upsert_credential(
    *,
    provider: str,
    user_id: UUID,
    provider_account_id: str,
    token_payload: dict[str, Any],
    account_label: Optional[str] = None,
    auth_type: str = "oauth2",
    scopes: Optional[str] = None,
    provider_metadata: Optional[dict[str, Any]] = None,
    token_expires_at: Optional[datetime] = None,
) -> IntegrationCredential:
    """Insert or replace the credential for a ``(provider, provider_account_id)``.

    Keyed on the external account, not the user. A reconnect by the **same**
    user takes the existing row over (a token refresh, matching the
    providers' invalidate-on-reinstall behavior). A reconnect by a **different**
    user while the current one is still active raises
    :class:`CrossUserConflictError` rather than silently stealing the
    workspace — the original owner must disconnect (or the account be revoked)
    first.
    """
    ciphertext, nonce, encryption_version, key_id = encrypt_credentials(token_payload)

    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential).where(
                IntegrationCredential.provider == provider,
                IntegrationCredential.provider_account_id == provider_account_id,
            )
        )
        credential = result.scalar_one_or_none()

        if (
            credential is not None
            and credential.status == STATUS_ACTIVE
            and credential.user_id != user_id
        ):
            logger.warning(
                "Refused %s reconnect: account %s already active for user %s, not %s",
                provider,
                provider_account_id,
                credential.user_id,
                user_id,
            )
            raise CrossUserConflictError(provider_account_id)

        if credential is None:
            credential = IntegrationCredential(
                provider=provider, provider_account_id=provider_account_id
            )
            db.add(credential)

        credential.user_id = user_id
        credential.account_label = account_label
        credential.auth_type = auth_type
        credential.scopes = scopes
        credential.provider_metadata = provider_metadata
        credential.ciphertext = ciphertext
        credential.nonce = nonce
        credential.encryption_version = encryption_version
        credential.key_id = key_id
        credential.token_expires_at = token_expires_at
        credential.status = STATUS_ACTIVE
        credential.revoked_at = None

        await db.commit()
        await db.refresh(credential)
        return credential


async def get_credential_by_account(
    provider: str, provider_account_id: str
) -> Optional[IntegrationCredential]:
    """Resolve an inbound webhook's external account id back to its credential."""
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential).where(
                IntegrationCredential.provider == provider,
                IntegrationCredential.provider_account_id == provider_account_id,
            )
        )
        return result.scalar_one_or_none()


async def get_active_credential_for_user(
    user_id: UUID, provider: str
) -> Optional[IntegrationCredential]:
    """The user's active connection for a provider, for the Integrations UI.

    A user *can* hold more than one active connection for a provider (two
    Slack workspaces, both provider='slack', same user) — the schema allows
    it. Ordered newest-first so the choice is deterministic (the most recently
    connected wins) rather than DB-arbitrary.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential)
            .where(
                IntegrationCredential.user_id == user_id,
                IntegrationCredential.provider == provider,
                IntegrationCredential.status == STATUS_ACTIVE,
            )
            .order_by(IntegrationCredential.created_at.desc())
        )
        return result.scalars().first()


async def revoke_credential_by_account(provider: str, provider_account_id: str) -> bool:
    """Mark a connection revoked. Idempotent — providers retry webhooks and
    give no ordering guarantee between uninstall/revoke events, so revoking an
    already-revoked row is a silent no-op.

    Returns True when a row exists (already- or newly-revoked), False if none.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential).where(
                IntegrationCredential.provider == provider,
                IntegrationCredential.provider_account_id == provider_account_id,
            )
        )
        credential = result.scalar_one_or_none()

        if credential is None:
            logger.warning("Revoke for unknown %s account %s", provider, provider_account_id)
            return False

        if credential.status != STATUS_REVOKED:
            credential.status = STATUS_REVOKED
            credential.revoked_at = datetime.now(UTC)
            await db.commit()

        return True


async def update_provider_metadata(
    provider: str, provider_account_id: str, metadata_patch: dict[str, Any]
) -> Optional[IntegrationCredential]:
    """Merge ``metadata_patch`` into a credential's ``provider_metadata``.

    Deliberately separate from :func:`upsert_credential`: that function
    re-encrypts and replaces the whole token payload on every call (it's the
    OAuth-install path), which would be the wrong tool for a settings tweak
    like a channel allowlist that has nothing to do with the token. Returns
    ``None`` if no credential exists for that account — callers translate
    that into a 404, this module has no HTTP opinions of its own.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential).where(
                IntegrationCredential.provider == provider,
                IntegrationCredential.provider_account_id == provider_account_id,
            )
        )
        credential = result.scalar_one_or_none()
        if credential is None:
            return None

        credential.provider_metadata = {**(credential.provider_metadata or {}), **metadata_patch}
        await db.commit()
        await db.refresh(credential)
        return credential


def decrypt_token_payload(credential: IntegrationCredential) -> dict[str, Any]:
    """Decrypt a credential's token payload — call only at provider-API time."""
    return decrypt_credentials(
        credential.ciphertext,
        credential.nonce,
        credential.encryption_version,
        credential.key_id,
    )
