"""Sync Gmail through the SDK connector's history and tombstone path."""

import re
from hashlib import sha256

from cognee.modules.integrations.google import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

GMAIL_DATASET_PREFIX = "gmail"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


def dataset_name_for_account(email: str, account_id: str = "") -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", email or "").strip("_").lower()
    name = f"{GMAIL_DATASET_PREFIX}_{slug or 'account'}"
    return f"{name}_{sha256(account_id.encode()).hexdigest()[:10]}" if account_id else name


def _dataset_name(credential: IntegrationCredential) -> str:
    metadata = credential.provider_metadata or {}
    return dataset_name_for_account(
        str(metadata.get("email") or ""), str(credential.provider_account_id)
    )


async def sync_gmail(credential: IntegrationCredential) -> None:
    """Run an initial or manual sync and record its outcome."""
    from cognee.modules.integrations.credentials import record_sync_result

    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    try:
        status, counts = await _sync_source(credential, counts)
    except Exception:
        counts["failed"] = max(1, counts["failed"])
        await record_sync_result(credential, status=SYNC_STATUS_DEGRADED, counts=counts)
        raise
    await record_sync_result(credential, status=status, counts=counts)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.integrations.gmail.adapter import access_token_for
    from cognee.modules.users.methods import get_user

    labels = (credential.provider_metadata or {}).get("selected_label_ids", [])
    if labels == []:
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}
    if labels is not None and (
        not isinstance(labels, list)
        or not all(isinstance(label, str) and label for label in labels)
    ):
        raise ValueError("Gmail selection must be a list of label IDs or null")

    source_factory = ingestion.source_factory("gmail")
    access_token = await access_token_for(credential)
    service = ingestion.build_service("gmail", access_token)
    owner = await get_user(credential.user_id)
    source = source_factory(label_ids=labels, service=service)
    try:
        result = await remember(
            source,
            dataset_name=_dataset_name(credential),
            user=owner,
            write_disposition="merge",
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )
    finally:
        ingestion.add_source_counts(counts, source)
    if getattr(result, "status", None) == "errored":
        counts["failed"] += 1
        counts["failed_ingestion"] = 1
    return (
        SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK,
        counts,
    )
