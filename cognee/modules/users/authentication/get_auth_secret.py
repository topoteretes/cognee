"""Secrets that sign authentication tokens.

Each secret is read from its environment variable. When the variable is unset
or blank, a random secret is generated once per process instead of falling
back to a shared, publicly known default. A generated secret is only known to
this process, so anything it signs stops verifying after a restart and is
rejected by other processes. Deployments with more than one process must set
the variables explicitly.

Generation itself is silent: this module is imported by ``import cognee``, and
SDK, CLI and MCP users never verify tokens, so a warning there would be noise.
The API server calls ``resolve_auth_secrets()`` from its lifespan, which is the
one place a generated secret is reported.
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
    return secrets.token_urlsafe(64)


def _configured_secret(env_var: str) -> str:
    return os.getenv(env_var, "").strip()


def get_auth_secret(env_var: str) -> str:
    """Return the secret in ``env_var``, or a per-process random secret when it is unset.

    Never logs: see the module docstring. Use ``resolve_auth_secrets()`` to report
    generated secrets at server startup.
    """
    return _configured_secret(env_var) or _generated_secret(env_var)


def resolve_auth_secrets() -> None:
    """Resolve every auth secret and warn once per generated one.

    Called from the API server lifespan so operators learn at boot, not on the
    first login, which secrets were generated. Silent when every variable is set.
    """
    for env_var, consequence in AUTH_SECRETS.items():
        if _configured_secret(env_var):
            continue
        _generated_secret(env_var)
        logger.warning(
            "%s is not set. A random secret was generated for this server process. %s "
            "If you run more than one process (uvicorn workers, several replicas, or "
            "separate API and worker processes), set %s in your .env to the same "
            "securely generated secret for every process.",
            env_var,
            consequence,
            env_var,
        )
