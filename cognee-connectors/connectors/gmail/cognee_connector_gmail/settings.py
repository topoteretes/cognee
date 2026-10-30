"""Moved from cognee.modules.integrations.gmail.gmail_settings (cognee dev, 2026-10-26).

Settings for the Gmail OAuth grant.
"""

from pydantic_settings import SettingsConfigDict

from cognee_connector_google.settings import GoogleSettings


class GoogleGmailSettings(GoogleSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_GMAIL_", extra="ignore")


google_gmail_settings = GoogleGmailSettings()
