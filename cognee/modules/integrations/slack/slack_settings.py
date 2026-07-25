from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings


class SlackSettings(IntegrationSettings):
    """Configuration for the Slack integration.

    One Slack app is installed into many workspaces; these values identify
    that single app. Secrets default to empty strings rather than failing at
    import so that deployments without Slack configured (and unit tests)
    still boot — consumers call :func:`require` at use time instead, which
    fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`IntegrationSettings`; ``signing_secret`` is the one
    field Slack itself needs beyond that shared shape.
    """

    model_config = SettingsConfigDict(env_prefix="SLACK_", extra="ignore")

    # Keys every inbound request's X-Slack-Signature HMAC. Also keys the OAuth
    # state parameter — both are server-side-only uses of the same secret.
    # Env: SLACK_SIGNING_SECRET
    signing_secret: str = ""


slack_settings = SlackSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing Slack secret must never degrade into a signature check against
    an empty key or an OAuth exchange with blank credentials — both would
    fail in confusing, downstream ways instead of naming the actual problem.
    """
    value = getattr(slack_settings, field_name)
    if not value:
        env_name = f"SLACK_{field_name.upper()}"
        raise RuntimeError(f"{env_name} is not configured")
    return value
