from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings

DEFAULT_MAX_FILE_SIZE_MB = 50.0


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

    # Files larger than this many MB are left out of a sync and counted in
    # ``skipped_too_large`` (never truncated). 50 by default, on review of PR #6668 and after
    # Google Drive's size cap: text and PDF files are read whole to extract their text, so a
    # file's size is memory during the sync, and every edit of a large file re-extracts it on
    # the tenant's LLM key. Raise or lower it here; it must be a positive number, and any
    # other non-blank value stops the server at startup (fail fast). A stored file that is
    # now over the cap (it grew, or the cap is new or lower) is forgotten, like any skip.
    # Env: DROPBOX_MAX_FILE_SIZE_MB
    max_file_size_mb: float = Field(default=DEFAULT_MAX_FILE_SIZE_MB, gt=0, allow_inf_nan=False)

    @field_validator("max_file_size_mb", mode="before")
    @classmethod
    def _blank_means_the_default(cls, value: object) -> object:
        # A blank value (the template line uncommented as shipped, a compose ``${VAR:-}``)
        # is unset, so the default applies; it is not a parse error that stops the API server.
        if isinstance(value, str) and not value.strip():
            return DEFAULT_MAX_FILE_SIZE_MB
        return value


dropbox_settings = DropboxSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset."""
    value = getattr(dropbox_settings, field_name)
    if not value:
        raise RuntimeError(f"DROPBOX_{field_name.upper()} is not configured")
    return value
