"""The settings base every provider's settings class subclasses.

Moved from cognee.modules.integrations.base.IntegrationSettings (cognee dev,
2026-10-26), renamed ``ConnectorSettings``; ``require`` generalizes the
per-provider ``require()`` helpers the cognee settings modules each carried.
"""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConnectorSettings(BaseSettings):
    """Common OAuth2 app credentials every provider settings class shares.

    A concrete provider subclasses this with its own ``env_prefix`` (e.g.
    ``NOTION_``) and adds provider-specific fields (Slack's
    ``signing_secret``, for one). Fields default to empty rather than failing
    at import so deployments without the integration configured still boot;
    fail loudly at use time instead (see :func:`require`).

    ``extra="ignore"`` is the default here on purpose: a prefix like
    ``NOTION_`` also covers variables that are not settings of the
    integration (the SDK connector's own ``NOTION_API_KEY``), and
    pydantic-settings would otherwise refuse to load at all.
    """

    model_config = SettingsConfigDict(extra="ignore")

    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    frontend_base_url: str = ""


def env_var_name(settings: Any, field_name: str) -> str:
    """The environment variable ``field_name`` is read from: ``{env_prefix}{FIELD}``.

    ``env_prefix`` comes from ``settings.model_config``; an object without one
    (a test stand-in) gets the bare upper-cased field name.
    """
    config = getattr(settings, "model_config", None) or {}
    prefix = config.get("env_prefix", "") or ""
    return f"{prefix}{field_name.upper()}"


def require(settings: Any, field_name: str) -> str:
    """Return a settings value, refusing to proceed when it is unset.

    A missing secret must never degrade into a signature check against an
    empty key or an OAuth exchange with blank credentials — both would fail
    in confusing, downstream ways instead of naming the actual problem. The
    error names the environment variable (``env_prefix`` plus the upper-cased
    field name) so the fix is unambiguous.
    """
    value = getattr(settings, field_name)
    if not value:
        raise RuntimeError(f"{env_var_name(settings, field_name)} is not configured")
    return value


__all__ = ["ConnectorSettings", "env_var_name", "require"]
