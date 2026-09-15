"""Secrets that sign authentication tokens.

Each secret is read from its environment variable. When the variable is unset
or empty, a random secret is generated once per process instead of falling
back to a shared, publicly known default. Tokens signed with a generated
secret do not survive a restart and are not accepted by other replicas, so
production deployments must set the variables explicitly.
"""

import os
import secrets
from functools import lru_cache

from cognee.shared.logging_utils import get_logger

logger = get_logger()


@lru_cache
def _generated_secret(env_var: str) -> str:
    logger.warning(
        "%s is not set; using a random secret generated for this process. "
        "Tokens signed with it will not survive a restart and will not be accepted "
        "by other replicas. Set %s to a long random string in production.",
        env_var,
        env_var,
    )
    return secrets.token_urlsafe(64)


def get_auth_secret(env_var: str) -> str:
    """Return the secret in ``env_var``, or a per-process random secret when it is unset."""
    value = os.getenv(env_var, "").strip()
    if value:
        return value
    return _generated_secret(env_var)
