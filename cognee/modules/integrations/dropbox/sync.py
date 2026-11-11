"""Sync Dropbox through the SDK source, one source per selected folder.

Core owns OAuth, the folder selection and dataset ownership. The bundled connector owns
listing, extraction, cursors and delete tombstones.
"""

import asyncio

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.dropbox import client
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

DROPBOX_DATASET_PREFIX = "dropbox"
SELECTION_KEY = "selected_folder_ids"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


class NestedSelectionError(ValueError):
    """One selected folder is inside another, so its files would be two documents."""


def dataset_name_for_credential(credential: IntegrationCredential) -> str:
    return ingestion.dataset_name_for_credential(DROPBOX_DATASET_PREFIX, credential)


def _classify_error(error: BaseException) -> str:
    """The count key a failed folder is reported under, from the source's typed error."""
    from cognee.modules.integrations.credentials import CredentialInactiveError
    from cognee.tasks.ingestion.connectors.dropbox import (
        DropboxAccountMismatchError,
        DropboxAuthError,
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
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return "failed_ingestion"


def _under(path: str, folder: str) -> bool:
    return path.startswith(folder + "/")


def check_selection_shape(selected: object) -> list[str]:
    """The selection as sorted folder ids, or ValueError when it is not a list of ``id:...``."""
    from cognee.tasks.ingestion.connectors.dropbox import valid_folder_id

    if not isinstance(selected, list) or not all(valid_folder_id(folder) for folder in selected):
        raise ValueError("Dropbox selection must be a list of folder ids (id:...)")
    return sorted(set(selected))


async def refuse_nested_selection(
    credential: IntegrationCredential, access_token: str, folder_ids: list[str]
) -> None:
    """Raise NestedSelectionError when a selected folder sits inside another one."""
    paths = []
    for folder_id in folder_ids:
        await ingestion.require_active_credential(credential)
        meta = await client.get_folder(access_token, folder_id)
        if meta is not None:
            paths.append(str(meta.get("path_lower") or ""))
    if any(_under(inner, outer) for inner in paths for outer in paths):
        raise NestedSelectionError("A selected Dropbox folder is inside another selected folder")


async def sync_dropbox(credential: IntegrationCredential) -> None:
    """Run a manual or scheduled sync and record its outcome."""
    await ingestion.run_sync("dropbox", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.modules.integrations.dropbox.adapter import access_token_for

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

    def fresh_token() -> str:
        # Called by the source's client before every request, on dlt's worker thread. The
        # access token lives about four hours and a sync of several folders can outlast it,
        # so each request asks the credential store, which refreshes it near expiry.
        return asyncio.run_coroutine_threadsafe(access_token_for(credential), loop).result()

    def make_source(folder_id: str, resource_name: str, check_active):
        return source_factory(
            folder_id=folder_id,
            account_id=str(credential.provider_account_id),
            resource_name=resource_name,
            token=fresh_token,
            check_active=check_active,
        )

    retained = await ingestion.sync_scopes(
        "dropbox",
        credential,
        counts,
        scopes=folder_ids,
        dataset_name=dataset_name,
        make_source=make_source,
        classify_error=_classify_error,
    )
    if not counts["failed"]:
        await ingestion.retire_resources("dropbox", credential, dataset_name, retained)
    return (SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK), counts
