from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.google.google_settings import GoogleSettings
from cognee.modules.integrations.google.google_settings import require as _require


class GoogleDriveSettings(GoogleSettings):
    """Configuration for the Google Drive integration.

    One Google Cloud OAuth client is used by every connecting account; these
    values identify that single client. Secrets default to empty strings
    rather than failing at import so that deployments without Drive configured
    (and unit tests) still boot — consumers call :func:`require` at use time
    instead, which fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`IntegrationSettings`; the field below is the one thing
    this integration needs beyond that shape.

    The prefix is ``GOOGLE_DRIVE_`` rather than ``GOOGLE_`` because the latter
    is already taken by unrelated settings (the translation task's
    ``GOOGLE_PROJECT_ID``), and because a later Gmail or Calendar connector is
    a separate provider with its own scopes and its own consent screen: they
    share the Cloud project, not the configuration block.
    """

    model_config = SettingsConfigDict(env_prefix="GOOGLE_DRIVE_", extra="ignore")

    # Signs the OAuth state parameter. Unlike Slack, GitHub and Linear, this
    # secret has exactly one use: Google sends no signed webhooks on this
    # path, so there is no inbound HMAC to key. Env: GOOGLE_DRIVE_STATE_SECRET
    state_secret: str = ""


google_drive_settings = GoogleDriveSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing value must never degrade into a token exchange with empty client
    credentials or a state signed with an empty key — both would fail in
    confusing, downstream ways instead of naming the actual problem.
    """
    return _require(google_drive_settings, field_name, "GOOGLE_DRIVE_")
