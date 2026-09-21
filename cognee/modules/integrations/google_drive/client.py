"""Thin async client for the Google endpoints this integration needs.

Deliberately hand-rolled over ``aiohttp`` rather than built on
``google-api-python-client``: the GitHub adapter set the precedent that an
in-tree provider uses core dependencies only (see
:mod:`cognee.modules.integrations.registry`, whose docstring makes that a
rule), and the four calls below are the entire surface this connector needs.
Pulling a Google SDK in for them would also re-introduce the per-source-SDK
weight that connectors were moved out of core to avoid.

Error messages carry the operation name and HTTP status only — never the
access token, the refresh token, the authorization code, a file's contents,
or Google's response body. Google's own client libraries do the opposite
(``HttpError`` carries the full response), which is a second reason not to
use them here: every failure in this module is logged by a detached webhook
or background task, where a leaked body would end up in server logs.
"""

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

_TIMEOUT = aiohttp.ClientTimeout(total=60)

# Hard ceiling on what one file may pull into memory. Drive holds whatever
# anyone put there, including multi-gigabyte exports, and the caller's own
# character trim happens only after the bytes have already arrived — so the
# stop has to be here, mid-stream, not downstream.
MAX_FILE_BYTES = 1_000_000

_CHUNK_BYTES = 64 * 1024


class GoogleAuthError(RuntimeError):
    """A token exchange or refresh Google rejected.

    Carries Google's stable ``error`` code (``invalid_grant``,
    ``invalid_client``, …) and never its ``error_description``, which echoes
    request content back. ``invalid_grant`` is the one callers branch on: it
    means the user revoked access on Google's side, or the refresh token
    expired, and no retry will help.
    """

    def __init__(self, operation: str, code: str):
        self.code = code
        super().__init__(f"Google {operation} failed: {code}")


async def _token_request(operation: str, payload: dict[str, str]) -> dict[str, Any]:
    """POST to the token endpoint and return its JSON, or raise.

    Google reports token-endpoint failures with a non-2xx status *and* an
    ``error`` field, so both are checked: the status alone would miss a
    malformed success body, and the field alone would miss a gateway error.
    """
    async with (
        aiohttp.ClientSession(timeout=_TIMEOUT) as session,
        session.post(TOKEN_URL, data=payload) as response,
    ):
        status = response.status
        try:
            body: dict[str, Any] = await response.json()
        except Exception:  # noqa: BLE001 - any non-JSON body is handled the same way
            # A non-JSON body is a gateway or proxy error, never Google's own
            # error shape. "from None" is deliberate: the decode error can
            # quote the body it failed on, which is exactly what this module
            # exists to keep out of logs.
            raise GoogleAuthError(operation, f"http_{status}") from None

    error = body.get("error")
    if error or status != 200:
        raise GoogleAuthError(operation, str(error) if error else f"http_{status}")

    if not body.get("access_token"):
        raise GoogleAuthError(operation, "no_access_token")
    return body


async def exchange_code(code: str, *, client_id: str, client_secret: str, redirect_uri: str):
    """Trade an authorization code for the account's tokens."""
    return await _token_request(
        "code exchange",
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )


async def refresh_access_token(refresh_token: str, *, client_id: str, client_secret: str):
    """Mint a fresh access token from a stored refresh token.

    Google's response carries no new refresh token: the original one stays
    valid until the user revokes it, so callers keep the one they already
    hold rather than expecting a rotation.
    """
    return await _token_request(
        "token refresh",
        {
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        },
    )


async def revoke_token(token: str) -> None:
    """Best-effort revoke of one token and the grant behind it.

    Google revokes the whole grant the token belongs to, which is why this
    connector never asks for ``include_granted_scopes``: keeping Drive's
    grant separate from any future Gmail or Calendar grant is what makes
    disconnecting one of them survivable for the others.
    """
    async with (
        aiohttp.ClientSession(timeout=_TIMEOUT) as session,
        session.post(REVOKE_URL, data={"token": token}) as response,
    ):
        if response.status != 200:
            logger.warning("Google token revoke returned HTTP %s", response.status)


async def _authorized_get(
    operation: str, url: str, access_token: str, params: dict[str, str] | None = None
) -> dict[str, Any]:
    async with (
        aiohttp.ClientSession(timeout=_TIMEOUT) as session,
        session.get(
            url, params=params, headers={"Authorization": f"Bearer {access_token}"}
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"Google {operation} failed: HTTP {response.status}")
        return await response.json()


async def fetch_userinfo(access_token: str) -> dict[str, Any]:
    """Identify the account that just authorized, using its fresh token.

    Google's token response says nothing about *who* authorized, but the
    credential has to be keyed on a stable account id. Rather than verifying
    the ``id_token`` JWT by hand (fetching Google's JWKS, matching a key id,
    checking claims), the fresh access token is spent on one userinfo call:
    the token came back over TLS from an exchange this server initiated, so
    what the endpoint reports about it needs no second proof. Same move the
    Linear adapter makes for the same reason.

    Returns ``sub`` (stable account id), ``email``, and ``hd`` (the Workspace
    domain, absent on personal accounts).
    """
    return await _authorized_get("userinfo", USERINFO_URL, access_token)


async def list_files(access_token: str, page_token: str | None = None) -> dict[str, Any]:
    """One page of the account's Drive files, newest-modified first.

    ``trashed = false`` is the only filter: which files are worth indexing is
    a decision for the caller, which knows what it can render as text.
    """
    params = {
        "q": "trashed = false",
        "fields": "nextPageToken, files(id, name, mimeType, webViewLink, modifiedTime, size)",
        "orderBy": "modifiedTime desc",
        "pageSize": "100",
        # Shared-drive items the account can reach are included rather than
        # erroring out on. The corpus stays the default (the account's own
        # Drive plus what is shared with it), which is the right scope for a
        # dataset only that account can read.
        "includeItemsFromAllDrives": "true",
        "supportsAllDrives": "true",
    }
    if page_token:
        params["pageToken"] = page_token
    return await _authorized_get("file listing", DRIVE_FILES_URL, access_token, params)


async def export_file(access_token: str, file_id: str, mime_type: str) -> str:
    """Export a Google-native document (Doc, Sheet, Slide) as text.

    Native documents have no bytes to download; Drive renders them on demand,
    which is why this is a different call from :func:`download_file`.
    """
    return await _authorized_text(
        "file export",
        f"{DRIVE_FILES_URL}/{file_id}/export",
        access_token,
        {"mimeType": mime_type},
    )


async def download_file(access_token: str, file_id: str) -> str:
    """Download a stored file's bytes, decoded as UTF-8 text."""
    return await _authorized_text(
        "file download",
        f"{DRIVE_FILES_URL}/{file_id}",
        access_token,
        {"alt": "media", "supportsAllDrives": "true"},
    )


async def _authorized_text(
    operation: str, url: str, access_token: str, params: dict[str, str]
) -> str:
    """Fetch a file body as text, replacing bytes that are not valid UTF-8.

    Read in chunks and abandoned at :data:`MAX_FILE_BYTES` rather than pulled
    in whole: a caller that trims the result afterwards has already paid the
    memory for everything Drive sent. Truncation can cut a multi-byte
    character in half, which is what ``errors="replace"`` is for.

    The last chunk is sliced to the ceiling rather than appended and then
    noticed. Checking after the append lets the read overshoot by almost a
    whole chunk, which turns a stated ceiling into an approximate one and
    makes the constant a lie about the worst case.

    Errors name the operation and status but never the file id: ids appear in
    shareable URLs, and this runs where every failure is logged.
    """
    chunks: list[bytes] = []
    received = 0

    async with (
        aiohttp.ClientSession(timeout=_TIMEOUT) as session,
        session.get(
            url, params=params, headers={"Authorization": f"Bearer {access_token}"}
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"Google {operation} failed: HTTP {response.status}")

        async for chunk in response.content.iter_chunked(_CHUNK_BYTES):
            remaining = MAX_FILE_BYTES - received
            if remaining <= 0:
                # Landed exactly on the ceiling on a previous chunk, and the
                # stream still had more waiting. A chunk that lands exactly
                # on the boundary looks identical to a file that legitimately
                # ends there until this next pull proves otherwise — that is
                # what the extra iteration is for.
                logger.info(
                    "Google %s stopped at the %d byte ceiling; the file is indexed truncated",
                    operation,
                    MAX_FILE_BYTES,
                )
                break
            if len(chunk) > remaining:
                # This chunk itself carries bytes past the ceiling, so there
                # is no ambiguity: truncation is certain without waiting for
                # another pull.
                chunks.append(chunk[:remaining])
                logger.info(
                    "Google %s stopped at the %d byte ceiling; the file is indexed truncated",
                    operation,
                    MAX_FILE_BYTES,
                )
                break
            chunks.append(chunk)
            received += len(chunk)

    return b"".join(chunks).decode("utf-8", errors="replace")
