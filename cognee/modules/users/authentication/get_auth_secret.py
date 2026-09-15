"""Secrets that sign authentication tokens.

Each secret is read from its environment variable. When the variable is unset
or blank, a random secret is generated once per process instead of falling
back to a shared, publicly known default, and a warning is logged. A generated
secret is only known to this process, so anything it signs stops verifying
after a restart and is rejected by other processes. Deployments with more than
one process must set the variables explicitly.
"""

import os
import secrets
from functools import lru_cache

from cognee.shared.logging_utils import get_logger

logger = get_logger()

# What stops working when the secret is generated rather than configured.
AUTH_SECRETS = {
    "FASTAPI_USERS_JWT_SECRET": (
        "Tokens signed with it are valid only for this process and only until it restarts."
    ),
    "FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET": (
        "Password reset links stop working when it restarts."
    ),
    "FASTAPI_USERS_VERIFICATION_TOKEN_SECRET": (
        "Email verification links stop working when it restarts."
    ),
}


@lru_cache
def _generated_secret(env_var: str) -> str:
    logger.warning(
        "%s is not set. A random secret was generated for this server process. %s "
        "If you run more than one process (uvicorn workers, several replicas, or "
        "separate API and worker processes), set %s in your .env to the same long "
        "random string for every process.",
        env_var,
        AUTH_SECRETS[env_var],
        env_var,
    )
    return secrets.token_urlsafe(64)


def get_auth_secret(env_var: str) -> str:
    """Return the secret in ``env_var``, or a per-process random secret when it is unset."""
    value = os.getenv(env_var, "").strip()
    if value:
        return value
    return _generated_secret(env_var)


def resolve_auth_secrets() -> None:
    """Resolve every auth secret now so generated ones are reported at startup."""
    for env_var in AUTH_SECRETS:
        get_auth_secret(env_var)
