from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings


class DropboxSettings(IntegrationSettings):
    """Configuration for the Dropbox integration.

    One Dropbox app is authorized by many accounts; ``client_id`` is its app key and
    ``client_secret`` its app secret. Values default to empty so deployments without Dropbox
    configured still boot; each is checked when used (:func:`require`).
    """

    model_config = SettingsConfigDict(env_prefix="DROPBOX_", extra="ignore")

    # Signs the OAuth state parameter, the only thing binding a callback to the user who
    # started it. Env: DROPBOX_STATE_SECRET
    state_secret: str = ""


dropbox_settings = DropboxSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset."""
    value = getattr(dropbox_settings, field_name)
    if not value:
        raise RuntimeError(f"DROPBOX_{field_name.upper()} is not configured")
    return value
