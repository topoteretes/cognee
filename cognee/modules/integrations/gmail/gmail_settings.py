"""Settings for the Gmail OAuth grant."""

from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.google.google_settings import GoogleSettings


class GoogleGmailSettings(GoogleSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_GMAIL_", extra="ignore")


google_gmail_settings = GoogleGmailSettings()
