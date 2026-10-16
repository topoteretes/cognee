"""Google Calendar OAuth adapter."""

from typing import Any

from cognee.modules.integrations.google.oauth_integration import GoogleOAuthIntegration
from cognee.modules.integrations.google_calendar import client as calendar_client
from cognee.modules.integrations.google_calendar.google_calendar_settings import (
    GoogleCalendarOAuthSettings,
    google_calendar_settings,
)
from cognee.modules.integrations.google_calendar.sync import GOOGLE_CALENDAR_DATASET_PREFIX
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


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
        from cognee.modules.integrations.google_calendar.sync import sync_google_calendar

        await sync_google_calendar(credential)

    async def list_resources(self, credential: IntegrationCredential) -> list[dict[str, Any]]:
        token = await access_token_for(credential)
        calendars = await calendar_client.list_calendars(token)
        return [
            {
                "id": str(calendar["id"]),
                "name": str(
                    calendar.get("summaryOverride") or calendar.get("summary") or calendar["id"]
                ),
                "description": calendar.get("description"),
                "attributes": {
                    "primary": bool(calendar.get("primary")),
                    "access_role": calendar.get("accessRole"),
                },
            }
            for calendar in calendars
            if calendar.get("id")
        ]


async def access_token_for(credential: IntegrationCredential) -> str:
    return await GoogleCalendarIntegration().access_token(credential)
