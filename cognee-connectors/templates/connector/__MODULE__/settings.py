"""__DISPLAY_NAME__ settings, read from ``__ENV_PREFIX__*`` environment variables."""

from pydantic_settings import SettingsConfigDict

from cognee_connectors_core.settings import ConnectorSettings


class __CLASS__Settings(ConnectorSettings):
    """OAuth application credentials and provider-specific settings.

    Fields default to empty so a deployment without this connector configured
    still boots; ``cognee_connectors_core.settings.require`` fails loudly, naming
    the environment variable, at use time.
    """

    model_config = SettingsConfigDict(env_prefix="__ENV_PREFIX__", extra="ignore")

    # __AUTH_FIELDS__


__PROVIDER___settings = __CLASS__Settings()
