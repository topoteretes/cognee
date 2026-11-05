"""Sync Drive through the SDK source's incremental, deletion-aware path.

Core owns OAuth, resource selection and dataset ownership. The bundled
connector owns listing, document extraction, cursors and delete tombstones.
"""

import asyncio

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.google_drive import client
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.shared.logging_utils import get_logger

logger = get_logger("google_drive_sync")

GOOGLE_DRIVE_DATASET_PREFIX = "google_drive"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


def _dataset_name(credential: IntegrationCredential) -> str:
    return ingestion.dataset_name_for_credential(GOOGLE_DRIVE_DATASET_PREFIX, credential)


def _classify_error(error: BaseException) -> str:
    """The count key a failed folder is reported under, from the source's typed error."""
    from cognee.tasks.ingestion.connectors.google_drive import DriveAuthError

    # dlt and remember() wrap the source's error; follow the chain to its end.
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, DriveAuthError):
            return "failed_auth"
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return "failed_ingestion"


async def sync_drive(credential: IntegrationCredential) -> None:
    """Run an initial or manual sync and record its outcome."""
    await ingestion.run_sync("google_drive", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.modules.integrations.google_drive.adapter import access_token_for
    from cognee.tasks.ingestion.connectors.google_drive import TokenRenewalError

    selected = (credential.provider_metadata or {}).get("selected_folder_ids")
    if selected == []:
        await ingestion.retire_resources(
            "google_drive", credential, _dataset_name(credential), set()
        )
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}
    if selected is not None and (
        not isinstance(selected, list)
        or not all(isinstance(folder, str) and folder for folder in selected)
    ):
        raise ValueError("Drive selection must be a list of folder IDs or null")

    source_factory = ingestion.source_factory("google_drive")
    access_token = await access_token_for(credential)
    active = await ingestion.require_active_credential(credential)
    # The token and its expiry, swapped as one pair. Shared by every worker client, so one built
    # after a renewal starts with the new token.
    live = {"grant": (access_token, active.token_expires_at)}
    renewal_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()

    async def renew(stale_token):
        # All file workers share the connection; only one may refresh it at a time.
        async with renewal_lock:
            if live["grant"][0] != stale_token:
                return  # another worker already replaced the token this client was using
            token = await access_token_for(credential)
            current = await ingestion.require_active_credential(credential)
            if current.token_expires_at is None:
                raise TokenRenewalError("Google Drive token renewal returned no expiry")
            live["grant"] = (token, current.token_expires_at)

    def refresh_access_token(stale_token):
        # No timeout, as in ``extraction_checkpoint``: the renewal is one database read and
        # one Google call (bounded by its own timeout), so a stuck database is what would
        # hold this thread, and the sync cannot proceed without that database anyway.
        # Any failure is an auth cut for this folder, not a per-file extraction failure; a
        # disconnected connection then stops the sync at the next folder's
        # require_active_credential. A cancelled renewal (the loop shutting down) is one too.
        try:
            asyncio.run_coroutine_threadsafe(renew(stale_token), loop).result()
        except (Exception, asyncio.CancelledError) as error:
            # The class only: the message may carry token material.
            logger.warning("Google Drive token renewal failed (%s)", type(error).__name__)
            raise TokenRenewalError("Google Drive token renewal failed") from error
        return live["grant"]

    def service_factory():
        # Without a known expiry the client cannot renew; it serves its token until it ends.
        token, expires_at = live["grant"]
        renewable = expires_at is not None
        return ingestion.build_service(
            "google_drive",
            token,
            expires_at=expires_at,
            refresh_access_token=refresh_access_token if renewable else None,
        )

    service = service_factory()
    shared_drive_ids: set[str] = set()
    page_token = None
    while True:
        await ingestion.require_active_credential(credential)
        # The first token: this loop ends before any file is read, so it never outlives it.
        page = await client.list_drives(access_token, page_token)
        shared_drive_ids.update(
            str(drive["id"]) for drive in page.get("drives", []) or [] if drive.get("id")
        )
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    folder_ids = list(dict.fromkeys(selected or ["root", *sorted(shared_drive_ids)]))

    def make_source(folder_id: str, resource_name: str, check_active):
        source_kwargs = {
            "folder_id": folder_id,
            "resource_name": resource_name,
            "service": service,
            "service_factory": service_factory,
            "check_active": check_active,
        }
        if folder_id in shared_drive_ids:
            source_kwargs["shared_drive_id"] = folder_id
        return source_factory(**source_kwargs)

    retained = await ingestion.sync_scopes(
        "google_drive",
        credential,
        counts,
        scopes=folder_ids,
        dataset_name=_dataset_name(credential),
        make_source=make_source,
        classify_error=_classify_error,
    )
    if not counts["failed"]:
        await ingestion.retire_resources(
            "google_drive", credential, _dataset_name(credential), retained
        )
    return (
        SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK,
        counts,
    )
