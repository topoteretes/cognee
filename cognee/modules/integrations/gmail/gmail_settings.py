"""Settings for the Gmail OAuth grant."""

from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.google.google_settings import GoogleSettings
from cognee.modules.integrations.google.google_settings import require as _require


class GoogleGmailSettings(GoogleSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_GMAIL_", extra="ignore")


google_gmail_settings = GoogleGmailSettings()


def require(field_name: str) -> str:
    return _require(google_gmail_settings, field_name, "GOOGLE_GMAIL_")
