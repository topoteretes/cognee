from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings


class NotionSettings(IntegrationSettings):
    """Configuration for the Notion integration.

    One Notion public integration is authorized by many workspaces; these values
    identify that single integration. Secrets default to empty strings rather than
    failing at import so deployments without Notion configured (and unit tests)
    still boot; each value is checked when used (:func:`require`).

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url`` come
    from :class:`IntegrationSettings`.
    """

    # ``extra="ignore"`` matters: the SDK's own connector reads ``NOTION_API_KEY`` from the
    # same prefix, and that variable is not a setting of this integration.
    model_config = SettingsConfigDict(env_prefix="NOTION_", extra="ignore")

    # Signs the OAuth state parameter, the only thing binding a callback to the user
    # who started it. Notion sends no signed webhooks on this path.
    # Env: NOTION_STATE_SECRET
    state_secret: str = ""


notion_settings = NotionSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset."""
    value = getattr(notion_settings, field_name)
    if not value:
        raise RuntimeError(f"NOTION_{field_name.upper()} is not configured")
    return value
