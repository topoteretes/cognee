"""Sync Dropbox through the SDK source, one source per selected folder.

Core owns OAuth, the folder selection and dataset ownership. The bundled connector owns
listing, extraction, cursors and delete tombstones.
"""

import asyncio
import threading
from collections.abc import Callable
from datetime import datetime
from itertools import pairwise

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.dropbox import client
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

DROPBOX_DATASET_PREFIX = "dropbox"
SELECTION_KEY = "selected_folder_ids"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


class SelectedFolderNotFoundError(ValueError):
    """A selected folder id Dropbox cannot resolve (deleted, or not visible to the account)."""


class NestedSelectionError(ValueError):
    """One selected folder is inside another, so its files would be two documents."""


def dataset_name_for_credential(credential: IntegrationCredential) -> str:
    return ingestion.dataset_name_for_credential(DROPBOX_DATASET_PREFIX, credential)


def _classify_error(error: BaseException) -> str:
    """The count key a failed folder is reported under, from the source's typed error."""
    from cognee.modules.integrations.credentials import CredentialInactiveError
    from cognee.tasks.ingestion.connectors.dropbox import (
        DropboxAccountMismatchError,
        DropboxApiError,
        DropboxAuthError,
        DropboxFolderNotFoundError,
        DropboxRateLimitedError,
    )

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, DropboxAccountMismatchError):
            # Not a token that expired: reconnecting the same account will not fix it.
            return "failed_account"
        if isinstance(current, (DropboxAuthError, CredentialInactiveError)) or (
            isinstance(current, client.DropboxIntegrationError) and current.code == "invalid_grant"
        ):
            # A refused token, or the token bridge failing to renew it: reconnect.
            return "failed_auth"
        if isinstance(current, DropboxRateLimitedError):
            return "failed_rate_limit"
        if isinstance(current, DropboxFolderNotFoundError):
            # A selected folder never listed that Dropbox does not find holds no data here,
            # so it must not block retiring the deselected ones. The source counts it the same.
            return "failed_folder_not_found"
        if isinstance(current, DropboxApiError):
            # The source raises this only while listing, and counts it the same way.
            return "failed_listing"
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return "failed_ingestion"


def check_selection_shape(selected: object) -> list[str]:
    """The selection as sorted folder ids, or ValueError when it is not a list of ``id:...``."""
    from cognee.tasks.ingestion.connectors.dropbox import valid_folder_id

    if not isinstance(selected, list) or not all(valid_folder_id(folder) for folder in selected):
        raise ValueError("Dropbox selection must be a list of folder ids (id:...)")
    return sorted(set(selected))


def _has_path(meta: dict) -> bool:
    """Whether a folder's metadata carries the path the nesting check compares; one without
    (unmounted) cannot be placed in the tree."""
    return isinstance(meta.get("path_lower"), str) and bool(meta["path_lower"])


async def refuse_nested_selection(
    credential: IntegrationCredential,
    access_token: str,
    folder_ids: list[str],
    *,
    refuse_missing: bool = False,
) -> None:
    """Raise NestedSelectionError when a selected folder sits inside another one.

    One credential check and one HTTP session for the whole selection. Paths are compared
    sorted with a trailing slash: a folder inside another sorts right after it (or after a
    folder that is itself inside it), so checking each against its neighbour is enough. A
    folder without a path (unmounted) takes no part: its "/" would be a prefix of all.

    ``refuse_missing`` (when a selection is saved) raises SelectedFolderNotFoundError for an
    id Dropbox cannot resolve. A sync lets it through: the source counts it as gone.
    """
    await ingestion.require_active_credential(credential)
    metas = await client.get_folders(access_token, folder_ids)
    if refuse_missing and not all(meta is not None and _has_path(meta) for meta in metas):
        raise SelectedFolderNotFoundError(
            "A selected Dropbox folder could not be found; pick it again from the list"
        )
    paths = sorted(
        meta["path_lower"] + "/" for meta in metas if meta is not None and _has_path(meta)
    )
    if any(path.startswith(previous) for previous, path in pairwise(paths)):
        raise NestedSelectionError("A selected Dropbox folder is inside another selected folder")


class _TokenCache:
    """The token the source's client sends, kept on dlt's worker thread between requests.

    It is a ``TokenSource`` (see ``cognee.tasks.ingestion.connectors.dropbox``). The access
    token lives about four hours and a sync of several folders can outlast it. The cache asks
    the credential store again (``fetch``) only when the token it holds is within the refresh
    margin of its expiry, or after Dropbox refused it: ``invalidate`` (called by the client on
    a 401, before it retries that request once) passes the refused token to ``fetch``, which
    renews it unless the row already holds another (renewed, or a reconnect). So the token
    path reads the credential once per token lifetime. The host's active check
    (``check_active``) still reads the row before every request.
    """

    def __init__(self, fetch: Callable[[bytes | None], tuple[str, datetime | None, bytes]]):
        self._fetch = fetch
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: datetime | None = None
        self._ciphertext: bytes | None = None
        self._refused: bytes | None = None

    def __call__(self) -> str:
        from cognee.modules.integrations import credentials as credential_store
        from cognee.modules.integrations.dropbox.adapter import _now

        with self._lock:
            if self._token is None or credential_store.token_is_expiring(self._expires_at, _now()):
                self._token, self._expires_at, self._ciphertext = self._fetch_or_auth_error()
                self._refused = None
            return self._token

    def _fetch_or_auth_error(self) -> tuple[str, datetime | None, bytes]:
        from cognee.modules.integrations.dropbox import client
        from cognee.tasks.ingestion.connectors.dropbox import DropboxAuthError

        failure = None
        try:
            return self._fetch(self._refused)
        except client.DropboxIntegrationError as error:
            if error.code != "invalid_grant":
                raise
            failure = error
        # Dropbox refused the refresh token: the adapter has revoked the connection. Raised
        # as the source's auth error, so the run stops cleanly and keeps the rows it has.
        raise DropboxAuthError(
            f"Dropbox refused the stored refresh token ({failure.code}); reconnect the account"
        )

    def invalidate(self) -> None:
        with self._lock:
            self._refused = self._ciphertext
            self._token = None


async def sync_dropbox(credential: IntegrationCredential) -> None:
    """Run a manual or scheduled sync and record its outcome."""
    await ingestion.run_sync("dropbox", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.modules.integrations.dropbox.adapter import (
        access_token_and_expiry,
        access_token_for,
    )
    from cognee.modules.integrations.dropbox.dropbox_settings import dropbox_settings

    selected = (credential.provider_metadata or {}).get(SELECTION_KEY)
    dataset_name = dataset_name_for_credential(credential)
    folder_ids = check_selection_shape(selected)
    if not folder_ids:
        # Nothing selected: nothing is synced and every table this account synced goes.
        await ingestion.retire_resources("dropbox", credential, dataset_name, set())
        return SYNC_STATUS_OK, counts

    access_token = await access_token_for(credential)
    await refuse_nested_selection(credential, access_token, folder_ids)

    source_factory = ingestion.source_factory("dropbox")
    loop = asyncio.get_running_loop()
    fresh_token = _TokenCache(
        lambda refused: asyncio.run_coroutine_threadsafe(
            access_token_and_expiry(credential, rejected_ciphertext=refused), loop
        ).result()
    )

    sources: dict[str, object] = {}
    cut_scopes = 0

    def make_source(folder_id: str, resource_name: str, check_active):
        sources[folder_id] = source = source_factory(
            folder_id=folder_id,
            account_id=str(credential.provider_account_id),
            resource_name=resource_name,
            token=fresh_token,
            check_active=check_active,
            max_file_size_mb=dropbox_settings.max_file_size_mb,
        )
        return source

    def scope_stored(folder_id: str) -> None:
        # A folder its own source cut short (rate limit, a 401 after rows) once its listing
        # was done and files were worked on: it resumes next pass and holds only its own
        # table, so the cut does not block retiring the deselected folders. A cut before
        # any file was worked on stored nothing and still blocks.
        nonlocal cut_scopes
        stats = getattr(sources[folder_id], "cognee_sync_stats", {})
        if not stats.get("complete") and (
            stats.get("scanned") or stats.get("deleted") or stats.get("failed_budget")
        ):
            cut_scopes += 1

    retained = await ingestion.sync_scopes(
        "dropbox",
        credential,
        counts,
        scopes=folder_ids,
        dataset_name=dataset_name,
        make_source=make_source,
        classify_error=_classify_error,
        on_scope_stored=scope_stored,
    )
    # One table per folder, named from its id, and no legacy names: a file that keeps failing
    # (failed_retry, counted per file and on every pass) leaves its folder's table as it is,
    # so it must not keep a deselected folder's data in memory for good. Any failure that
    # stored nothing (a listing error, an account mismatch, a raised source) still blocks.
    if not ingestion.retirement_blocked(
        counts,
        gone_keys=("failed_retry", "failed_folder_gone", "failed_folder_not_found"),
        cut_scopes=cut_scopes,
    ):
        await ingestion.retire_resources("dropbox", credential, dataset_name, retained)
    return (SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK), counts
