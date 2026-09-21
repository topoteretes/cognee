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
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.crypto import decrypt_credentials, encrypt_credentials
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_REVOKED = "revoked"


class _Unset:
    """Marker for an argument :func:`upsert_credential` was not given.

    ``None`` is a meaningful value for both ``workspace_id`` (no workspace
    owns this row) and ``provider_metadata`` (nothing stored), so "not
    supplied" has to be distinguishable from it. Without that distinction
    every caller that leaves one out silently clears what is already there.
    """

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


class CrossUserConflictError(Exception):
    """A different owner already holds an active connection for this external account.

    An external account maps to exactly one owner (UNIQUE(provider,
    provider_account_id)). Silently reassigning it on reconnect would let one
    owner take over another's connection with no trace — so we refuse and let
    the original owner disconnect first.

    "Owner" is ``workspace_id`` when the caller passes one to
    :func:`upsert_credential`, otherwise ``user_id`` — see that function.
    """


async def upsert_credential(
    *,
    provider: str,
    user_id: UUID,
    provider_account_id: str,
    token_payload: dict[str, Any],
    workspace_id: UUID | None | _Unset = UNSET,
    account_label: str | None = None,
    auth_type: str = "oauth2",
    scopes: str | None = None,
    provider_metadata: dict[str, Any] | None | _Unset = UNSET,
    token_expires_at: datetime | None = None,
) -> IntegrationCredential:
    """Insert or replace the credential for a ``(provider, provider_account_id)``.

    Keyed on the external account, not the owner. A reconnect by the **same**
    owner takes the existing row over (a token refresh, matching the
    providers' invalidate-on-reinstall behavior). A reconnect by a **different**
    owner while the current one is still active raises
    :class:`CrossUserConflictError` rather than silently stealing the
    connection — the original owner must disconnect (or the account be
    revoked) first.

    ``workspace_id`` is optional. Omitted, it reproduces the original
    single-user contract exactly: ``user_id`` is the owner, conflicts are
    compared on it, and whatever the row already carries is left untouched.
    Pass ``workspace_id`` when several users share one connection (a
    cognee-hosted multi-user workspace, or a downstream layer's own tenant
    concept) and it becomes the owner/conflict key instead, while ``user_id``
    still records who connected it.

    ``provider_metadata`` is **merged** into what is stored rather than
    swapping for it. Providers rebuild their whole metadata dict from each
    token response, so every key they own is refreshed either way; what
    merging protects is the keys that cannot be re-derived from that response
    because they were written separately through
    :func:`update_provider_metadata` (today: Slack's channel allowlist).
    Replacing instead of merging meant a plain reconnect through
    :mod:`cognee.modules.integrations.connect` dropped them silently, and an
    empty Slack allowlist reads as "every channel allowed".

    Three consequences of merging worth knowing, none of them accidental.
    Neither ``None`` nor ``{}`` clears the stored metadata, so this function
    cannot empty it at all — unlike ``workspace_id``, where an explicit
    ``None`` still means "no workspace owns this". Keys an adapter stops
    emitting now survive, so an adapter that builds its dict conditionally
    has to write the key with a null rather than leave it out. And a revoked
    row is reused rather than replaced (see below), so a new owner connecting
    a previously-connected account inherits its metadata, which for the
    allowlist means starting restricted rather than open.

    The other optional arguments (``account_label``, ``auth_type``,
    ``scopes``, ``token_expires_at``) are still assigned unconditionally: a
    caller that omits one clears it. That is the pre-existing contract and
    every current caller passes what it needs, but do not read "omitted means
    untouched" as applying to them.
    """
    # Resolved before the conflict check so that an omitted workspace_id is
    # read as "this caller does not deal in workspaces" (the original
    # user-owned contract) rather than as an explicit None, while still
    # leaving any stored workspace_id alone further down.
    conflict_workspace_id = None if isinstance(workspace_id, _Unset) else workspace_id

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

        if credential is not None and credential.status == STATUS_ACTIVE:
            existing_owner = (
                credential.workspace_id if conflict_workspace_id is not None else credential.user_id
            )
            new_owner = conflict_workspace_id if conflict_workspace_id is not None else user_id
            if existing_owner != new_owner:
                logger.warning(
                    "Refused %s reconnect: account %s already active for owner %s, not %s",
                    provider,
                    provider_account_id,
                    existing_owner,
                    new_owner,
                )
                raise CrossUserConflictError(provider_account_id)

        if credential is None:
            credential = IntegrationCredential(
                provider=provider, provider_account_id=provider_account_id
            )
            db.add(credential)

        credential.user_id = user_id
        if not isinstance(workspace_id, _Unset):
            credential.workspace_id = workspace_id
        credential.account_label = account_label
        credential.auth_type = auth_type
        credential.scopes = scopes
        if not isinstance(provider_metadata, _Unset):
            # Best-effort under concurrency: this read-modify-write and
            # update_provider_metadata's both run unlocked in separate
            # transactions, so a reconnect racing a settings write can still
            # lose one of them. Both windows are a single statement wide.
            credential.provider_metadata = {
                **(credential.provider_metadata or {}),
                **(provider_metadata or {}),
            }
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
) -> IntegrationCredential | None:
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
) -> IntegrationCredential | None:
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


async def list_active_credentials_for_user(user_id: UUID) -> dict[str, IntegrationCredential]:
    """The user's active connections across all providers, one query.

    Returns a dict of ``provider -> credential`` holding at most one
    credential per provider, chosen newest-first — the same tiebreak as
    :func:`get_active_credential_for_user`, so the aggregate status endpoint
    and the per-provider connection endpoint always agree on which
    connection represents a provider. For display use: token material stays
    encrypted on the rows and must not be surfaced by callers.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential)
            .where(
                IntegrationCredential.user_id == user_id,
                IntegrationCredential.status == STATUS_ACTIVE,
            )
            .order_by(IntegrationCredential.created_at.desc())
        )
        credentials: dict[str, IntegrationCredential] = {}
        for credential in result.scalars().all():
            credentials.setdefault(credential.provider, credential)
        return credentials


async def get_active_credential_for_workspace(
    workspace_id: UUID, provider: str
) -> IntegrationCredential | None:
    """The workspace's active connection for a provider.

    Mirrors :func:`get_active_credential_for_user` for the ``workspace_id``
    owner dimension — use this instead when the connection was upserted with
    a ``workspace_id`` (several users sharing one connection), since
    ``user_id`` on that row is only who connected it, not the owner.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as db:
        result = await db.execute(
            select(IntegrationCredential)
            .where(
                IntegrationCredential.workspace_id == workspace_id,
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
            credential.revoked_at = datetime.now(timezone.utc)
            await db.commit()

        return True


async def update_provider_metadata(
    provider: str, provider_account_id: str, metadata_patch: dict[str, Any]
) -> IntegrationCredential | None:
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


async def record_sync_result(
    credential: IntegrationCredential,
    *,
    status: str,
    scanned: int | None = None,
    skipped: int | None = None,
    failed: int | None = None,
    counts: dict[str, int] | None = None,
) -> IntegrationCredential | None:
    """Stamp when a connector last synced a connection and how it went.

    ``last_synced_at`` and ``sync_status`` have existed on the row since the
    table was created and nothing wrote them, which left a failed or partial
    sync invisible: the connection still read as healthy while its memory was
    empty. A connector that has no webhook to self-heal on has no other way to
    say so, which is why it stamps the outcome here.

    Takes the connection the sync actually ran for rather than an external
    account id, and re-checks it before writing. Syncs run detached and can
    outlive the install that started them, while
    :func:`upsert_credential` **reuses** the row for a
    ``(provider, provider_account_id)`` rather than replacing it. So a stamp
    addressed to the account alone would land on whoever holds that account
    now: disconnect, someone else connects the same Google account, the old
    owner's sync finishes last, and the new owner is told their Drive just
    synced. The row is written only while it is still active and still owned
    by the same owner (``workspace_id`` when set, ``user_id`` otherwise —
    the same resolution :func:`upsert_credential` uses); a token refresh
    mid-sync keeps both, so the ordinary path is unaffected.

    Not closed by this check: the **same** owner disconnecting and
    reconnecting the same account can still have a stale sync from the old
    install stamp the new one. Nothing distinguishes one install of an
    account from the next without a generation column, which this table
    does not have.

    ``status`` is the vocabulary the integrations UI already renders,
    ``"ok"`` or ``"degraded"``. Deliberately separate from
    :func:`upsert_credential`, which re-encrypts the whole token payload: a
    sync result has nothing to do with the token. Best-effort by contract —
    the caller is a detached background task and a failed stamp must never
    take down a sync that otherwise worked.
    """
    try:
        # Read from the object inside the guard, not above it: this can run
        # from inside an ``except`` block (see ``sync_drive``), and anything
        # raising here — even attribute access on a detached instance — must
        # not replace the traceback that is already in flight.
        provider = credential.provider
        provider_account_id = credential.provider_account_id
        # Mirrors upsert_credential's owner resolution exactly: workspace_id
        # is the owner when the caller set one, user_id otherwise. Comparing
        # user_id alone would miss the same misattribution on the dimension
        # this check is supposed to cover — a workspace-scoped connection
        # reconnected under a different workspace by the same human.
        ran_for_owner = (
            credential.workspace_id if credential.workspace_id is not None else credential.user_id
        )

        engine = get_relational_engine()
        async with engine.get_async_session() as db:
            result = await db.execute(
                select(IntegrationCredential).where(
                    IntegrationCredential.provider == provider,
                    IntegrationCredential.provider_account_id == provider_account_id,
                )
            )
            current = result.scalar_one_or_none()
            if current is None:
                return None

            current_owner = (
                current.workspace_id if current.workspace_id is not None else current.user_id
            )
            if current.status != STATUS_ACTIVE or current_owner != ran_for_owner:
                logger.info(
                    "Discarding a %s sync result for account %s: the connection it ran for "
                    "is gone (status %s, owner %s, was %s)",
                    provider,
                    provider_account_id,
                    current.status,
                    current_owner,
                    ran_for_owner,
                )
                return None

            current.last_synced_at = datetime.now(timezone.utc)
            current.sync_status = status
            if counts is not None or any(value is not None for value in (scanned, skipped, failed)):
                current.provider_metadata = {
                    **(current.provider_metadata or {}),
                    "last_sync_counts": {
                        "scanned": scanned if scanned is not None else 0,
                        "skipped": skipped if skipped is not None else 0,
                        "failed": failed if failed is not None else 0,
                        **(counts or {}),
                    },
                }
            await db.commit()
            await db.refresh(current)
            return current
    except Exception:
        logger.exception(
            "Recording the sync result for %s account %s failed", provider, provider_account_id
        )
        return None


def decrypt_token_payload(credential: IntegrationCredential) -> dict[str, Any]:
    """Decrypt a credential's token payload — call only at provider-API time."""
    return decrypt_credentials(
        credential.ciphertext,
        credential.nonce,
        credential.encryption_version,
        credential.key_id,
    )
