"""Slack settings, read from ``SLACK_*`` environment variables.

Copied from cognee.modules.integrations.slack.slack_settings (cognee dev, 2026-11-12):
``IntegrationSettings`` became :class:`cognee_connectors_core.settings.ConnectorSettings`
and :func:`require` delegates to the core helper of the same name, so the environment
variable names and the error a missing value raises are unchanged.
"""

from pydantic_settings import SettingsConfigDict

from cognee_connectors_core.settings import ConnectorSettings
from cognee_connectors_core.settings import require as _require


class SlackSettings(ConnectorSettings):
    """Configuration for the Slack integration.

    One Slack app is installed into many workspaces; these values identify
    that single app. Secrets default to empty strings rather than failing at
    import so that deployments without Slack configured (and unit tests)
    still boot — consumers call :func:`require` at use time instead, which
    fails loudly per missing value.

    ``client_id``/``client_secret``/``redirect_uri``/``frontend_base_url``
    come from :class:`ConnectorSettings`; ``signing_secret`` is the one
    field Slack itself needs beyond that shared shape.
    """

    model_config = SettingsConfigDict(env_prefix="SLACK_", extra="ignore")

    # Keys every inbound request's X-Slack-Signature HMAC. Also keys the OAuth
    # state parameter — both are server-side-only uses of the same secret.
    # Env: SLACK_SIGNING_SECRET
    signing_secret: str = ""

    # Where Slack sends a member back after Sign in with Slack (member_link.py):
    # the generic integrations callback of the ``slack_member`` provider,
    # https://<host>/api/v1/integrations/slack_member/callback. It must be one of
    # the app's registered redirect URLs. Env: SLACK_MEMBER_REDIRECT_URI
    member_redirect_uri: str = ""

    # The in-process tick that syncs every connection's selected channels every
    # six hours (see sync.py). Disable it when an external scheduler calls
    # sync_due_workspaces instead. The uninstall cleanup
    # (forget_uninstalled_workspaces) runs either way. Env: SLACK_HISTORY_SYNC_ENABLED
    history_sync_enabled: bool = True


slack_settings = SlackSettings()


def require(field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing Slack secret must never degrade into a signature check against
    an empty key or an OAuth exchange with blank credentials — both would
    fail in confusing, downstream ways instead of naming the actual problem.
    """
    return _require(slack_settings, field_name)
