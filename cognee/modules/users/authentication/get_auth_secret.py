import os
import secrets
from functools import lru_cache

from cognee.shared.logging_utils import get_logger

logger = get_logger()

JWT_SECRET_ENV_VAR = "FASTAPI_USERS_JWT_SECRET"
RESET_PASSWORD_TOKEN_SECRET_ENV_VAR = "FASTAPI_USERS_RESET_PASSWORD_TOKEN_SECRET"
VERIFICATION_TOKEN_SECRET_ENV_VAR = "FASTAPI_USERS_VERIFICATION_TOKEN_SECRET"


@lru_cache
def get_auth_secret(env_var: str) -> str:
    """Return the auth secret named by ``env_var``, or a random per-process one.

    ``.env.template`` already describes the shipped default as INSECURE and tells
    operators to override it, but an unset variable still resolved to a literal
    that is identical in every cognee installation, so the signing key of a
    default deployment was public knowledge.

    When the variable is unset (or empty) this returns a random secret instead.
    It is deliberately **not** stable across restarts or across instances: tokens
    minted before a restart stop verifying, and instances do not accept each
    other's tokens. That is the failure the shared constant was hiding, and it
    is the signal to set the variable — which ``.env.template`` documents as
    required for multi-instance deployments regardless.

    Results are cached per variable name, so one process resolves each secret
    once and every caller of that variable agrees within the process. ``env_var``
    is deliberately required rather than defaulted: ``lru_cache`` keys on the
    call signature, so a defaulted argument would let ``get_auth_secret()`` and
    ``get_auth_secret(JWT_SECRET_ENV_VAR)`` generate two different secrets for
    one variable, and a token signed under one would not verify under the other.
    """
    secret = os.getenv(env_var)
    if secret:
        return secret

    logger.warning(
        "%s is not set. Falling back to a random secret generated for this process: "
        "tokens will not survive a restart and will not be accepted by other instances. "
        "Set %s to a long random string to issue stable tokens.",
        env_var,
        env_var,
    )
    return secrets.token_urlsafe(64)
