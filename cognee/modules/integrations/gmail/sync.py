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
    await ingestion.run_sync("gmail", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.integrations.gmail.adapter import access_token_for
    from cognee.modules.users.methods import get_user

    labels = (credential.provider_metadata or {}).get("selected_label_ids", [])
    if labels == []:
        await ingestion.retire_resources("gmail", credential, _dataset_name(credential), set())
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
    resource_name = ingestion.resource_name("gmail", credential)
    source = source_factory(
        label_ids=labels,
        service=service,
        resource_name=resource_name,
        check_active=ingestion.extraction_checkpoint(credential),
    )
    await ingestion.require_active_credential(credential)
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
    else:
        await ingestion.retire_resources(
            "gmail", credential, _dataset_name(credential), {resource_name}
        )
    return (
        SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK,
        counts,
    )
