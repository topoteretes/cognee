from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings


class LinearSettings(IntegrationSettings):
    """Configuration for the Linear agent integration.

    One Linear OAuth app is installed into many workspaces; these values
    identify that single app. Secrets default to empty strings rather than
    failing at import so that deployments without Linear configured (and
    unit tests) still boot — consumers call :func:`require` at use time
    instead, which fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`IntegrationSettings`; the field below is the one thing
    a Linear agent app needs beyond that shape.
    """

    model_config = SettingsConfigDict(env_prefix="LINEAR_", extra="ignore")

    # Keys every inbound delivery's linear-signature HMAC. Also keys the
    # OAuth state parameter — both are server-side-only uses of the same
    # secret (same doubling-up as Slack's signing_secret and GitHub's
    # webhook_secret). Env: LINEAR_WEBHOOK_SECRET
    webhook_secret: str = ""


linear_settings = LinearSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing Linear secret must never degrade into a signature check
    against an empty key or a token exchange with empty app credentials —
    both would fail in confusing, downstream ways instead of naming the
    actual problem.
    """
    value = getattr(linear_settings, field_name)
    if not value:
        env_name = f"LINEAR_{field_name.upper()}"
        raise RuntimeError(f"{env_name} is not configured")
    return value
