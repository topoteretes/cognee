"""Moved from cognee.modules.integrations.google_calendar.adapter (cognee dev, 2026-10-26).

Google Calendar OAuth adapter.
"""

from typing import Any

from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

from cognee_connector_google.cognee_host.oauth_integration import GoogleOAuthIntegration
from cognee_connector_google_calendar import resources
from cognee_connector_google_calendar.cognee_host.sync import GOOGLE_CALENDAR_DATASET_PREFIX
from cognee_connector_google_calendar.settings import (
    GoogleCalendarOAuthSettings,
    google_calendar_settings,
)


class GoogleCalendarIntegration(GoogleOAuthIntegration):
    provider = "google_calendar"
    settings_cls = GoogleCalendarOAuthSettings
    settings = google_calendar_settings
    # Opt-in by default: events carry attendees and meeting details, so the user
    # must choose calendars before sync.
    resource_selection_key = "selected_calendar_ids"
    # The connector reads a null selection as the primary calendar, or as the
    # server-wide GOOGLE_CALENDAR_IDS, so "all" has no safe meaning for a user.
    resource_selection_allows_all = False
    display_name = "Google Calendar"
    scope = "openid email https://www.googleapis.com/auth/calendar.readonly"
    dataset_prefix = GOOGLE_CALENDAR_DATASET_PREFIX

    async def sync_now(self, credential: IntegrationCredential) -> None:
        from cognee_connector_google_calendar.cognee_host.sync import sync_google_calendar

        await sync_google_calendar(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        return await resources.list_resources(await access_token_for(credential))


async def access_token_for(credential: IntegrationCredential) -> str:
    return await GoogleCalendarIntegration().access_token(credential)
