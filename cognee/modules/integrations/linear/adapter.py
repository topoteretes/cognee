"""Linear (agent app) as an ``OAuthIntegration`` adapter.

Connects a Linear workspace through an *agent* install rather than a plain
OAuth consent: the authorize URL carries ``actor=app``, which installs an
app user — the agent's identity — into the workspace, and the
``app:assignable``/``app:mentionable`` scopes let members delegate issues to
it or @mention it. Those mentions/delegations arrive as agent session
webhooks and are answered from cognee memory
(:mod:`cognee.modules.integrations.linear.agent_session`).

The token exchange differs from the plain-OAuth shape in one way, absorbed
by ``exchange_callback``: Linear's token response says nothing about *which*
workspace authorized the app, but every webhook delivery routes by its
``organizationId`` envelope field — so the fresh token is immediately spent
on one GraphQL query for ``viewer`` (the app user) and ``organization``, and
the merged result is what ``parse_installation`` (which is sync, so it
cannot make that call itself) turns into a credential keyed on the
organization id.

``revoke_remote`` is a real implementation here, unlike GitHub's deliberate
no-op: Linear exposes a cheap token-revoke endpoint that kills only *our*
token, whereas GitHub's remote equivalent would be uninstalling the app from
the whole org, which is too destructive for a cognee-side disconnect.

``refresh`` is a real implementation as well, because Linear access tokens
last 24 hours. Unlike Google, Linear rotates the refresh token: every refresh
returns a new one and the one just spent stops working, so the stored token
is replaced on each refresh instead of being carried forward.
``access_token_for`` is the only sanctioned path to a bearer token. It
refreshes an expiring token, one refresh at a time per credential.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import aiohttp
from sqlalchemy.exc import SQLAlchemyError

from cognee.infrastructure.background_tasks import register_background_task
from cognee.modules.integrations.base import (
    OAuthInstallation,
    OAuthIntegration,
    WebhookVerifier,
)
from cognee.modules.integrations.credentials import (
    CredentialInactiveError,
    decrypt_token_payload,
    require_active_credential,
    revoke_credential_if_current,
    update_refreshed_credential,
)
from cognee.modules.integrations.linear.client import graphql
from cognee.modules.integrations.linear.handle_linear_event import handle_linear_event
from cognee.modules.integrations.linear.linear_settings import LinearSettings, require
from cognee.modules.integrations.linear.sync import sync_recent_issues
from cognee.modules.integrations.linear.verify_linear_signature import LinearWebhookVerifier
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_AUTHORIZE_URL = "https://linear.app/oauth/authorize"
_TOKEN_URL = "https://api.linear.app/oauth/token"
_REVOKE_URL = "https://api.linear.app/oauth/revoke"

# app:assignable lets members delegate issues to the agent; app:mentionable
# lets them @mention it. Both require the actor=app install (and actor=app
# cannot be combined with the admin scope).
_SCOPES = "read,write,app:assignable,app:mentionable"

_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Two revokes run one after the other during a disconnect, so each gets a short one.
_REVOKE_TIMEOUT = aiohttp.ClientTimeout(total=10)

# A refresh sits in front of the first agent activity, and Linear wants that
# activity within 10 seconds of a session being created. So a refresh gets far
# less than the general timeout above.
_REFRESH_TIMEOUT = aiohttp.ClientTimeout(total=5)

# Refresh this far before the token actually dies, so a sync that takes a
# while does not expire halfway through its own issue list.
_REFRESH_MARGIN = timedelta(minutes=5)
_DEFAULT_EXPIRES_IN = 86399
_MAX_EXPIRES_IN = 30 * 24 * 3600  # anything longer is not a token lifetime Linear gives

# A refresh that times out or fails to save may still have rotated the token at
# Linear. Linear replays the same request for 30 minutes, so one more attempt
# well inside that window recovers it.
_RETRY_DELAY = 60

# How long an invalid_grant waits before it revokes. A second process that won
# the race for the same refresh token needs a moment to commit its new one.
_INVALID_GRANT_SETTLE = 3

# After a transient refresh failure, callers within this many seconds fail (or
# use the still-valid token) at once instead of each running a refresh of their own.
_FAILURE_MEMORY = 5

# viewer is the freshly installed app user (the agent identity in that
# workspace); organization.id is what every webhook envelope routes by.
_INSTALL_CONTEXT_QUERY = """
query InstallContext {
  viewer { id name }
  organization { id name urlKey }
}
"""

# One lock per credential row, so that callers who find the same token expiring
# do not each spend the same refresh token. Process-local, like the session
# locks: a second server process is covered by the compare-and-swap in
# ``update_refreshed_credential``, not by this. Entries are never expired, which
# is fine for one row per connected workspace.
_refresh_locks: dict[UUID, asyncio.Lock] = {}

# Credentials with a retry already waiting, and the tasks anchoring them.
_pending_retries: set[UUID] = set()
_retry_tasks: set[asyncio.Task] = set()

# When the last transient refresh failure of a credential happened, and what it was.
_recent_failures: dict[UUID, tuple[float, Exception]] = {}


# The OAuth error codes worth telling apart. Anything else, including text a
# proxy or WAF put in the field, is reported by its HTTP status instead.
_ERROR_CODES = frozenset(
    {
        "invalid_grant",
        "invalid_client",
        "invalid_request",
        "invalid_scope",
        "unauthorized_client",
        "unsupported_grant_type",
        "access_denied",
        "server_error",
        "temporarily_unavailable",
        "Error",
    }
)


class LinearAuthError(RuntimeError):
    """A stable, log-safe failure of Linear's token endpoint."""

    def __init__(self, operation: str, code: str):
        self.code = code
        super().__init__(f"Linear {operation} failed: {code}")


async def refresh_access_token(
    refresh_token: str, *, client_id: str, client_secret: str
) -> dict[str, Any]:
    """Spend a refresh token on a new access token and a new refresh token.

    Linear takes this as a form-encoded POST and answers a rejected token with
    HTTP 400 and ``{"error": "invalid_grant"}``. The error is reduced to that
    code so neither the request nor the response body can reach a log.
    """
    async with (
        aiohttp.ClientSession(timeout=_REFRESH_TIMEOUT) as session,
        session.post(
            _TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
            },
        ) as response,
    ):
        status = response.status
        try:
            body: dict[str, Any] = await response.json()
        except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
            raise LinearAuthError("token refresh", f"http_{status}") from None

    if not isinstance(body, dict):
        raise LinearAuthError("token refresh", f"http_{status}")
    # Only a short code-like string is kept. A proxy or WAF can answer with any
    # JSON, and its text must not become part of an exception that gets logged.
    error = body.get("error")
    if error or status != 200:
        code = error if isinstance(error, str) and error in _ERROR_CODES else None
        raise LinearAuthError("token refresh", code or f"http_{status}")
    if not body.get("access_token"):
        raise LinearAuthError("token refresh", "no_access_token")
    return body


def _expires_at(expires_in: Any) -> datetime | None:
    if not expires_in:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))


def _lifetime(expires_in: Any) -> int:
    """Seconds a refreshed token lives, falling back to a day for anything unusable.

    Runs after Linear has already rotated the token, so a response it cannot
    read must not stop the new tokens from being saved.
    """
    if isinstance(expires_in, bool):
        return _DEFAULT_EXPIRES_IN
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError, OverflowError):  # OverflowError: inf
        return _DEFAULT_EXPIRES_IN
    return seconds if 0 < seconds <= _MAX_EXPIRES_IN else _DEFAULT_EXPIRES_IN


def _expires_within(credential: IntegrationCredential, margin: timedelta) -> bool:
    expires_at = credential.token_expires_at
    if expires_at is None:
        return False
    # A naive timestamp comes back from SQLite, which stores no timezone; it
    # is written as UTC, so that is what it is read as.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at - margin <= datetime.now(timezone.utc)


def _is_expiring(credential: IntegrationCredential) -> bool:
    return _expires_within(credential, _REFRESH_MARGIN)


def _keep_valid_token(credential: IntegrationCredential, error: Exception) -> IntegrationCredential:
    """Fall back to the stored token when a refresh failed but it still works.

    The margin refreshes early, so for a few minutes the old token is valid
    while the token endpoint may not be. Failing there would fail sessions the
    old token could have answered. Past its real expiry there is nothing to
    fall back to and the error is raised.
    """
    if _expires_within(credential, timedelta(0)):
        raise error
    logger.warning(
        "Linear token refresh for organization %s failed (%s); using the token that is still valid",
        credential.provider_account_id,
        error.__class__.__name__,
    )
    return credential


async def access_token_for(credential: IntegrationCredential) -> str:
    """A usable access token for this credential, refreshed if it is about to die.

    The only sanctioned path from a stored credential to a usable bearer
    token. Call it only at Linear-API time and again for every unit of work
    rather than holding the result, since the token lasts 24 hours. It raises
    rather than returning an empty string so a malformed payload fails loudly
    at the call site instead of as a Linear 401.
    """
    credential = await require_active_credential(credential)
    if _is_expiring(credential):
        credential = await _refresh_expiring(credential)

    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise RuntimeError(
            f"Linear credential for organization {credential.provider_account_id} "
            f"holds no access token"
        )
    return token


def _retry_refresh_later(credential: IntegrationCredential) -> None:
    """Try the refresh once more in a minute, detached, for a workspace nobody is waiting on."""
    if credential.id in _pending_retries:
        return
    _pending_retries.add(credential.id)

    async def _attempt() -> None:
        try:
            await _refresh_expiring(credential, retry=False)
        except CredentialInactiveError:
            logger.info(
                "Linear token refresh retry for organization %s dropped: connection is gone",
                credential.provider_account_id,
            )
        except Exception:  # detached work must log, not crash the loop
            logger.exception(
                "Linear token refresh retry for organization %s failed",
                credential.provider_account_id,
            )

    async def _retry() -> None:
        try:
            await asyncio.sleep(_RETRY_DELAY)
            # Only the refresh itself is registered for the shutdown drain, not
            # the minute of waiting: the drain gives up after a few seconds, so
            # waiting on the sleep would only delay shutdown. Once the request is
            # out, Linear may rotate the token, and the drain should let the save
            # finish. The shield keeps it running if this task is cancelled.
            await asyncio.shield(register_background_task(asyncio.create_task(_attempt())))
        finally:
            _pending_retries.discard(credential.id)

    task = asyncio.create_task(_retry())
    _retry_tasks.add(task)
    task.add_done_callback(_retry_tasks.discard)


def _transient_failure(
    credential: IntegrationCredential, error: Exception, retry: bool
) -> IntegrationCredential:
    """Handle a refresh that failed in a way that may have followed a rotation.

    Linear may have rotated the token before the answer was lost or the save
    failed. The row still holds the spent one, which is only good for Linear's
    replay window, so do not wait for the next caller: it may come hours later
    on a quiet workspace.
    """
    _recent_failures[credential.id] = (time.monotonic(), error)
    if retry:
        _retry_refresh_later(credential)
    return _keep_valid_token(credential, error)


async def _refresh_expiring(
    credential: IntegrationCredential, *, retry: bool = True
) -> IntegrationCredential:
    """Refresh under the credential's lock and return the row as it is afterwards."""
    lock = _refresh_locks.setdefault(credential.id, asyncio.Lock())
    if lock.locked() and not _expires_within(credential, timedelta(0)):
        # Someone else is refreshing and this token still works. Waiting would
        # only queue behind a refresh that can take its full timeout.
        return credential

    async with lock:
        # Whoever held the lock may have refreshed already. Going on with the
        # row we were handed would spend a refresh token that rotation has
        # since killed.
        credential = await require_active_credential(credential)
        if not _is_expiring(credential):
            return credential

        # The retry exists to try again, so it never answers from the memory.
        recent = _recent_failures.get(credential.id) if retry else None
        if recent and time.monotonic() - recent[0] < _FAILURE_MEMORY:
            return _keep_valid_token(credential, recent[1])

        rejected: LinearAuthError | None = None
        try:
            await LinearIntegration().refresh(credential)
        except CredentialInactiveError:
            # The row changed while the refresh was in flight. The read below
            # tells which way: a disconnected or replaced connection raises
            # there, while a refresh by another process leaves a usable token.
            pass
        except LinearAuthError as error:
            if error.code == "invalid_grant":
                # refresh() has already revoked the credential if it was still
                # the current one. If it was not, another process refreshed
                # first and the read below returns that token.
                rejected = error
            else:
                # A 5xx from a gateway, or a body that never finished, can come
                # after Linear rotated the token. See the next branch.
                return _transient_failure(credential, error, retry)
        except (aiohttp.ClientError, asyncio.TimeoutError, SQLAlchemyError) as error:
            return _transient_failure(credential, error, retry)
        except asyncio.CancelledError:
            if retry:
                _retry_refresh_later(credential)
            raise
        else:
            _recent_failures.pop(credential.id, None)

        # refresh() writes through its own session, so the instance we were
        # handed still carries the pre-rotation ciphertext. Read the row back
        # rather than decrypting a stale one.
        try:
            return await require_active_credential(credential)
        except CredentialInactiveError as inactive:
            raise inactive from rejected


class LinearIntegration(OAuthIntegration):
    provider = "linear"
    settings_cls = LinearSettings

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": require("client_id"),
            "redirect_uri": require("redirect_uri"),
            "response_type": "code",
            "state": state,
            # The agent install: puts an app user into the workspace instead
            # of acting as the authorizing human.
            "actor": "app",
            "scope": _SCOPES,
        }
        return f"{_AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange the OAuth code for the workspace's agent token."""
        async with (
            aiohttp.ClientSession(timeout=_TIMEOUT) as session,
            session.post(
                _TOKEN_URL,
                data={
                    "code": code,
                    "redirect_uri": require("redirect_uri"),
                    "client_id": require("client_id"),
                    "client_secret": require("client_secret"),
                    "grant_type": "authorization_code",
                },
            ) as response,
        ):
            if response.status != 200:
                raise RuntimeError(f"Linear code exchange failed: HTTP {response.status}")
            payload: dict[str, Any] = await response.json()

        if not payload.get("access_token"):
            raise RuntimeError("Linear code exchange returned no access_token")
        return payload

    async def exchange_callback(self, code: str, params: dict[str, str]) -> dict[str, Any]:
        """Exchange the code, then enrich the response with workspace identity.

        The raw token response carries no workspace or app-user identity, but
        ``parse_installation`` is sync and must derive ``provider_account_id``
        from this response alone — and that id must be the organization id,
        because webhooks route back by their ``organizationId`` envelope
        field. So the fresh token is spent here, in the async leg, on one
        GraphQL query and the result rides along.
        """
        token_response = await self.exchange_code(code)
        install_context = await graphql(token_response["access_token"], _INSTALL_CONTEXT_QUERY)
        return {
            **token_response,
            "viewer": install_context.get("viewer"),
            "organization": install_context.get("organization"),
        }

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        organization = token_response.get("organization") or {}
        organization_id = organization.get("id")
        if not organization_id:
            raise ValueError("Linear token response carries no organization id")

        # Secret material stays in token_payload (encrypted at rest). The
        # refresh token is what keeps the connection alive past the first 24
        # hours (see ``refresh``).
        token_payload = {"access_token": token_response["access_token"]}
        if token_response.get("refresh_token"):
            token_payload["refresh_token"] = token_response["refresh_token"]

        viewer = token_response.get("viewer") or {}
        return OAuthInstallation(
            provider_account_id=str(organization_id),
            token_payload=token_payload,
            provider_metadata={
                "app_user_id": viewer.get("id"),
                "app_user_name": viewer.get("name"),
                "organization_name": organization.get("name"),
                "organization_url_key": organization.get("urlKey"),
                "scope": token_response.get("scope"),
            },
            account_label=organization.get("name"),
            scopes=token_response.get("scope"),
            token_expires_at=_expires_at(token_response.get("expires_in")),
            auth_type="oauth2",
        )

    def state_signing_secret(self) -> str:
        return require("webhook_secret")

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    def webhook_verifier(self) -> WebhookVerifier | None:
        return LinearWebhookVerifier()

    async def handle_webhook(self, raw_body: bytes, headers: dict[str, str]) -> None:
        await handle_linear_event(raw_body, headers)

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Seed memory with the workspace's recently active issues.

        Issue webhooks only cover changes from now on — and any delivery
        racing ahead of the OAuth callback storing the credential is dropped
        as unknown (same race as GitHub's ``installation.created``). This
        hook, firing after the upsert, is what gives the agent something to
        recall from on day one.
        """
        await sync_recent_issues(credential)

    async def sync_now(self, credential: IntegrationCredential) -> None:
        await sync_recent_issues(credential)

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort remote revoke of the workspace's agent token.

        Cheap and non-destructive on Linear's side (it kills only this
        token, not the app install), so worth doing — but still best-effort:
        the local revoke is the actual access cut-off, and a network blip
        here must never block a disconnect.
        """
        # The stored tokens themselves, not ``access_token_for``: by the time
        # someone disconnects the access token has usually expired, and a
        # refresh right before a revoke would only add a way for this to fail.
        # Linear documents the ``token`` field as the form to use (the
        # Authorization header is accepted only for backwards compatibility),
        # but not whether revoking the refresh token also ends the access
        # token, so both are revoked, refresh token first.
        try:
            token_payload = decrypt_token_payload(credential)
        except Exception:  # disconnect must proceed no matter what happens here
            logger.exception(
                "Linear token revoke for organization %s failed", credential.provider_account_id
            )
            return
        for token_type in ("refresh_token", "access_token"):
            token = token_payload.get(token_type)
            if not token:
                continue
            try:
                async with (
                    aiohttp.ClientSession(timeout=_REVOKE_TIMEOUT) as session,
                    session.post(
                        _REVOKE_URL, data={"token": token, "token_type_hint": token_type}
                    ) as response,
                ):
                    if response.status != 200:
                        logger.warning(
                            "Linear %s revoke for organization %s failed: HTTP %s",
                            token_type,
                            credential.provider_account_id,
                            response.status,
                        )
            except Exception:  # one failed revoke must not stop the other or the disconnect
                logger.exception(
                    "Linear %s revoke for organization %s failed",
                    token_type,
                    credential.provider_account_id,
                )

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Rotate the access token in place, and the refresh token with it.

        Linear returns a new refresh token with every refresh and the one just
        spent stops working, so the response's token replaces the stored one.
        Carrying the old one forward, as the Google adapters do, would make the
        second refresh fail and disconnect the workspace about two days after
        it connected.

        An ``invalid_grant`` revokes the local credential. Linear tells us
        about an uninstall through the ``OAuthApp`` revoked webhook, so this is
        more likely a rotation that was lost than a user's decision. Either way
        the stored refresh token is dead and only a reconnect brings it back,
        so keeping the connection would only make it look healthy while every
        agent session fails.
        """
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(
                f"Linear credential for organization {credential.provider_account_id} "
                f"holds no refresh token; the workspace must reconnect"
            )

        try:
            refreshed = await refresh_access_token(
                refresh_token,
                client_id=require("client_id"),
                client_secret=require("client_secret"),
            )
        except LinearAuthError as error:
            if error.code == "invalid_grant":
                # Another process may have spent the same refresh token a
                # moment ago and be about to store the new one. The revoke only
                # fires if the row is still the one we read, so give that
                # process time to commit first.
                await asyncio.sleep(_INVALID_GRANT_SETTLE)
                await revoke_credential_if_current(credential)
                logger.warning(
                    "Linear rejected the refresh token for organization %s; "
                    "local credential revoked, the workspace must reconnect",
                    credential.provider_account_id,
                )
            raise

        new_refresh_token = refreshed.get("refresh_token")
        if not new_refresh_token:
            logger.warning(
                "Linear refresh for organization %s returned no refresh token; keeping the old one",
                credential.provider_account_id,
            )
        await update_refreshed_credential(
            credential,
            token_payload={
                "access_token": refreshed["access_token"],
                # Linear documents a new refresh token on every refresh. A response
                # without one is off contract, and the old token is the only one
                # left to try.
                "refresh_token": new_refresh_token or refresh_token,
            },
            # Linear documents expires_in. Without one, assume the 24 hours it
            # gives, since a missing expiry would mean never refreshing again.
            token_expires_at=_expires_at(_lifetime(refreshed.get("expires_in"))),
            scopes=refreshed.get("scope") or credential.scopes,
        )
