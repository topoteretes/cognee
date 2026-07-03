"""Helpers for loading authentication secrets safely.

Historically the JWT / token secrets fell back to the hardcoded literal
``"super_secret"`` when their environment variables were unset. That meant any
deployment that forgot to configure them was signing tokens with a public,
well-known key — allowing anyone to forge a valid JWT for any user (including a
superuser) and fully bypass authentication.

``get_auth_secret`` fixes this by refusing to start with an insecure default in
a production environment. In non-production environments (``ENV`` != ``prod``)
it falls back to a development default and logs a loud warning so local work is
not disrupted.
"""

import os

from cognee.shared.logging_utils import get_logger

logger = get_logger("auth_secret")

# The old, publicly-known default. Treated as "not configured".
_INSECURE_DEFAULT = "super_secret"

# Development fallback used only when ENV is not production.
_DEV_DEFAULT = "insecure-dev-secret-do-not-use-in-production"


def _is_production() -> bool:
    # Mirrors cognee.api.client: ENV defaults to "prod".
    return os.getenv("ENV", "prod").lower() == "prod"


def get_auth_secret(env_var: str) -> str:
    """Return the configured secret for ``env_var``.

    Raises ``RuntimeError`` in production when the secret is missing or set to a
    known-insecure placeholder. Falls back to a dev-only value otherwise.
    """
    secret = os.getenv(env_var)

    if secret and secret != _INSECURE_DEFAULT:
        return secret

    if _is_production():
        raise RuntimeError(
            f"{env_var} is not set (or uses the insecure default). Set a strong, "
            f"random secret before running in production. Generate one with: "
            f"python -c 'import secrets; print(secrets.token_urlsafe(64))'"
        )

    logger.warning(
        "%s is not set — using an insecure development default. "
        "This MUST be set to a strong random value in production.",
        env_var,
    )
    return _DEV_DEFAULT
