from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.google.google_settings import GoogleSettings


class GoogleDriveSettings(GoogleSettings):
    """Configuration for the Google Drive integration.

    One Google Cloud OAuth client is used by every connecting account; these
    values identify that single client. Secrets default to empty strings
    rather than failing at import so that deployments without Drive configured
    (and unit tests) still boot; each value is checked when used
    (``GoogleOAuthIntegration._require``), which fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`IntegrationSettings`.

    The prefix is ``GOOGLE_DRIVE_`` rather than ``GOOGLE_`` because the latter
    is already taken by unrelated settings (the translation task's
    ``GOOGLE_PROJECT_ID``), and because a later Gmail or Calendar connector is
    a separate provider with its own scopes and its own consent screen: they
    share the Cloud project, not the configuration block.
    """

    model_config = SettingsConfigDict(env_prefix="GOOGLE_DRIVE_", extra="ignore")


google_drive_settings = GoogleDriveSettings()
