from pydantic_settings import SettingsConfigDict

from cognee.modules.integrations.base import IntegrationSettings


class GithubSettings(IntegrationSettings):
    """Configuration for the GitHub App integration.

    One GitHub App is installed into many orgs/accounts; these values
    identify that single app. Secrets default to empty strings rather than
    failing at import so that deployments without GitHub configured (and
    unit tests) still boot — consumers call :func:`require` at use time
    instead, which fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`IntegrationSettings`; the fields below are what a
    GitHub *App* (as opposed to a plain OAuth app) needs beyond that shape.
    """

    model_config = SettingsConfigDict(env_prefix="GITHUB_", extra="ignore")

    # The app's numeric id, from the app settings page. Issues the app JWT
    # that mints installation tokens. Env: GITHUB_APP_ID
    app_id: str = ""

    # The app's URL slug (github.com/apps/<slug>), used to build the
    # installation URL the connect flow redirects to. Env: GITHUB_APP_SLUG
    app_slug: str = ""

    # The app's RS256 private key, PEM-encoded. Accepts literal "\n" escapes
    # so it survives single-line env files. Env: GITHUB_APP_PRIVATE_KEY
    app_private_key: str = ""

    # Keys every inbound delivery's X-Hub-Signature-256 HMAC. Also keys the
    # OAuth state parameter — both are server-side-only uses of the same
    # secret (same doubling-up as Slack's signing_secret).
    # Env: GITHUB_WEBHOOK_SECRET
    webhook_secret: str = ""


github_settings = GithubSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing GitHub secret must never degrade into a signature check
    against an empty key or a JWT signed with an empty PEM — both would fail
    in confusing, downstream ways instead of naming the actual problem.
    """
    value = getattr(github_settings, field_name)
    if not value:
        env_name = f"GITHUB_{field_name.upper()}"
        raise RuntimeError(f"{env_name} is not configured")
    return value


def private_key_pem() -> str:
    """The app private key with env-file "\\n" escapes restored to newlines."""
    return require("app_private_key").replace("\\n", "\n")
