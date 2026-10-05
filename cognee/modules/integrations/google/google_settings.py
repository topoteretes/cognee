"""Settings shared by Google OAuth integrations."""

from typing import Any

from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings


class GoogleSettings(IntegrationSettings):
    """Common Google OAuth application settings."""

    model_config = SettingsConfigDict(extra="ignore")

    state_secret: str = ""


def require(settings: GoogleSettings, field_name: str, env_prefix: str) -> str:
    """Return a configured setting or fail with the exact environment name."""
    value: Any = getattr(settings, field_name)
    if not value:
        raise RuntimeError(f"{env_prefix}{field_name.upper()} is not configured")
    return value
