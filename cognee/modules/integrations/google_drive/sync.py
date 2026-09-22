"""Sync Drive through the SDK source's incremental, deletion-aware path.

Core owns OAuth, resource selection and dataset ownership. The bundled
connector owns listing, document extraction, cursors and delete tombstones.
"""

import logging
import re
from hashlib import sha256

from cognee.modules.integrations.google import ingestion
from cognee.modules.integrations.google_drive import client
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

GOOGLE_DRIVE_DATASET_PREFIX = "google_drive"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


def dataset_name_for_account(email: str, account_id: str = "") -> str:
    """Keep the account's dataset stable and disambiguate slug collisions."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", email or "").strip("_").lower()
    name = f"{GOOGLE_DRIVE_DATASET_PREFIX}_{slug or 'account'}"
    return f"{name}_{sha256(account_id.encode()).hexdigest()[:10]}" if account_id else name


def _dataset_name(credential: IntegrationCredential) -> str:
    email = (credential.provider_metadata or {}).get("email") or ""
    return dataset_name_for_account(email, str(credential.provider_account_id))


async def sync_drive(credential: IntegrationCredential) -> None:
    """Run an initial or manual sync and record its outcome."""
    await ingestion.run_sync("google_drive", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.integrations.google_drive.adapter import access_token_for
    from cognee.modules.users.methods import get_user

    selected = (credential.provider_metadata or {}).get("selected_folder_ids")
    if selected == []:
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}
    if selected is not None and (
        not isinstance(selected, list)
        or not all(isinstance(folder, str) and folder for folder in selected)
    ):
        raise ValueError("Drive selection must be a list of folder IDs or null")

    source_factory = ingestion.source_factory("google_drive")
    access_token = await access_token_for(credential)
    service = ingestion.build_service("google_drive", access_token)
    shared_drive_ids: set[str] = set()
    page_token = None
    while True:
        page = await client.list_drives(access_token, page_token)
        shared_drive_ids.update(
            str(drive["id"]) for drive in page.get("drives", []) or [] if drive.get("id")
        )
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    folder_ids = list(dict.fromkeys(selected or ["root", *sorted(shared_drive_ids)]))
    owner = await get_user(credential.user_id)
    for folder_id in folder_ids:
        source = None
        try:
            source_kwargs = {
                "folder_id": folder_id,
                "resource_name": (
                    f"google_drive_files_{sha256(folder_id.encode()).hexdigest()[:16]}"
                ),
                "service": service,
            }
            if folder_id in shared_drive_ids:
                source_kwargs["shared_drive_id"] = folder_id
            source = source_factory(**source_kwargs)
            result = await remember(
                source,
                dataset_name=_dataset_name(credential),
                user=owner,
                write_disposition="merge",
                primary_key="id",
                max_rows_per_table=0,
                self_improvement=False,
            )
            if getattr(result, "status", None) == "errored":
                counts["failed"] += 1
                counts["failed_ingestion"] = counts.get("failed_ingestion", 0) + 1
        except Exception:
            counts["failed"] += 1
            counts["failed_ingestion"] = counts.get("failed_ingestion", 0) + 1
            logger.exception(
                "Google Drive sync failed for account %s folder %s",
                credential.provider_account_id,
                folder_id,
            )
        finally:
            ingestion.add_source_counts(counts, source)
    return (
        SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK,
        counts,
    )
