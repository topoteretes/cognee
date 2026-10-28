"""Linear settings, read from ``LINEAR_*`` environment variables.

Moved from cognee.modules.integrations.linear.linear_settings (cognee dev, 2026-10-26).
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from cognee_connectors_core.settings import ConnectorSettings
from cognee_connectors_core.settings import require as require_setting


class LinearSettings(ConnectorSettings):
    """Configuration for the Linear agent integration.

    One Linear OAuth app is installed into many workspaces; these values
    identify that single app. Secrets default to empty strings rather than
    failing at import so that deployments without Linear configured (and
    unit tests) still boot — consumers call :func:`require` at use time
    instead, which fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`ConnectorSettings`; the field below is the one thing
    a Linear agent app needs beyond that shape.
    """

    model_config = SettingsConfigDict(env_prefix="LINEAR_", extra="ignore")

    # Keys every inbound delivery's linear-signature HMAC. Also keys the
    # OAuth state parameter — both are server-side-only uses of the same
    # secret (same doubling-up as Slack's signing_secret and GitHub's
    # webhook_secret). Env: LINEAR_WEBHOOK_SECRET
    webhook_secret: str = ""

    # A first sync of a big workspace outlasts one hour of Linear's quota, so a
    # run stops cleanly and has to be continued. This worker continues a
    # connection whose last run was cut short by the rate limit or the request
    # budget. Env: LINEAR_RESUME_SYNC_ENABLED, LINEAR_RESUME_INTERVAL_SECONDS
    resume_sync_enabled: bool = True
    resume_interval_seconds: int = Field(default=600, ge=60)


linear_settings = LinearSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing Linear secret must never degrade into a signature check
    against an empty key or a token exchange with empty app credentials —
    both would fail in confusing, downstream ways instead of naming the
    actual problem. The error names the environment variable
    (``LINEAR_<FIELD>``), through the core's
    :func:`cognee_connectors_core.settings.require`.
    """
    return require_setting(linear_settings, field_name)
