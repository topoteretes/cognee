"""Settings for the Google Calendar OAuth grant.

The OAuth module needs only the application settings. The SDK connector's
``GoogleCalendarSettings`` adds the local-sync fields (refresh token, credential
files, calendar ids) and reads the same ``GOOGLE_CALENDAR_*`` block; hosted syncs
never use those fields, so the connector class is not imported here.
"""

from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.google.google_settings import GoogleSettings


class GoogleCalendarOAuthSettings(GoogleSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_CALENDAR_", extra="ignore")


google_calendar_settings = GoogleCalendarOAuthSettings()
