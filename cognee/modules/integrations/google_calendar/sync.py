"""Sync Google Calendar through the SDK connector's sync-token and tombstone path."""

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

GOOGLE_CALENDAR_DATASET_PREFIX = "google_calendar"


def _dataset_name(credential: IntegrationCredential) -> str:
    return ingestion.dataset_name_for_credential(GOOGLE_CALENDAR_DATASET_PREFIX, credential)


async def sync_google_calendar(credential: IntegrationCredential) -> None:
    """Run an initial or manual sync and record its outcome."""
    await ingestion.run_sync("google_calendar", credential, _sync_source)


async def _sync_source(
    credential: IntegrationCredential, counts: dict[str, int]
) -> tuple[str, dict[str, int]]:
    from cognee.modules.integrations.google_calendar.adapter import (
        GoogleCalendarIntegration,
        access_token_for,
    )

    selection_key = GoogleCalendarIntegration.resource_selection_key
    calendars = (credential.provider_metadata or {}).get(selection_key, [])
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
    # One source for all selected calendars: the connector keeps a sync token per
    # calendar in this resource's state and forgets the events of a calendar that
    # was deselected, so a changed selection needs no per-calendar resource here.
    return await ingestion.sync_selected_resources(
        "google_calendar",
        credential,
        counts,
        selection=calendars,
        selection_kwarg="calendar_ids",
        dataset_name=_dataset_name(credential),
        access_token_for=access_token_for,
    )
