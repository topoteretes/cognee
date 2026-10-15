"""Sync Google Calendar through the SDK connector's sync-token and tombstone path."""

import re
from hashlib import sha256

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

GOOGLE_CALENDAR_DATASET_PREFIX = "google_calendar"
SYNC_STATUS_OK = "ok"
SYNC_STATUS_DEGRADED = "degraded"


def dataset_name_for_account(email: str, account_id: str = "") -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", email or "").strip("_").lower()
    name = f"{GOOGLE_CALENDAR_DATASET_PREFIX}_{slug or 'account'}"
    return f"{name}_{sha256(account_id.encode()).hexdigest()[:10]}" if account_id else name


def _dataset_name(credential: IntegrationCredential) -> str:
    metadata = credential.provider_metadata or {}
    return dataset_name_for_account(
        str(metadata.get("email") or ""), str(credential.provider_account_id)
    )


async def sync_google_calendar(credential: IntegrationCredential) -> None:
    """Run an initial or manual sync and record its outcome."""
    await ingestion.run_sync("google_calendar", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.api.v1.remember.remember import remember
    from cognee.modules.integrations.google_calendar.adapter import (
        GoogleCalendarIntegration,
        access_token_for,
    )
    from cognee.modules.users.methods import get_user

    selection_key = GoogleCalendarIntegration.resource_selection_key
    calendars = (credential.provider_metadata or {}).get(selection_key, [])
    if calendars == []:
        await ingestion.retire_resources(
            "google_calendar", credential, _dataset_name(credential), set()
        )
        return SYNC_STATUS_OK, {"scanned": 0, "skipped": 0, "failed": 0}
    # Unlike the generic selection contract, null cannot mean "all calendars" here:
    # the connector reads calendar_ids=None as the primary calendar or whatever
    # GOOGLE_CALENDAR_IDS holds in the server environment, which would sync one
    # deployment-wide list for every user. Hosted syncs never reach that default.
    if not isinstance(calendars, list) or not all(
        isinstance(calendar, str) and calendar.strip() for calendar in calendars
    ):
        raise ValueError(
            "Calendar selection must be a list of calendar IDs; null is not supported "
            "because Google Calendar syncs only calendars chosen explicitly"
        )

    source_factory = ingestion.source_factory("google_calendar")
    access_token = await access_token_for(credential)
    service = ingestion.build_service("google_calendar", access_token)
    owner = await get_user(credential.user_id)
    resource_name = ingestion.resource_name("google_calendar", credential)
    # One source for all selected calendars: the connector keeps a sync token per
    # calendar in this resource's state and forgets the events of a calendar that
    # was deselected, so a changed selection needs no per-calendar resource here.
    source = source_factory(
        calendar_ids=calendars,
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
            "google_calendar", credential, _dataset_name(credential), {resource_name}
        )
    return (
        SYNC_STATUS_DEGRADED if counts["failed"] else SYNC_STATUS_OK,
        counts,
    )
