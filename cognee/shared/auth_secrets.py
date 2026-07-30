"""Centralized secret resolution for authentication tokens.

All auth-related secrets (JWT signing, password reset, email verification)
flow through :func:`resolve_secret`.  When the configured value is the
well-known insecure placeholder ``"super_secret"`` (the default shipped in
``.env.template``), a cryptographically random key is generated instead so
that a default deployment cannot be exploited with a publicly known signing
key (#4265).

The generated key is stable for the lifetime of the process.  For
multi-instance deployments (e.g. multiple Kubernetes pods that must accept
each other's tokens), operators **must** set the corresponding environment
variable to a long random string — the per-process random fallback is only
suitable for single-instance development use.
"""

import logging
import os
import secrets

logger = logging.getLogger(__name__)

# The insecure placeholder shipped in .env.template.  Any secret still holding
# this value at resolution time is replaced with a random key.
_INSECURE_PLACEHOLDER = "super_secret"

# Cache generated fallbacks per env-var name so repeated lookups within a
# process return the same key (tokens signed earlier remain verifiable).
_generated_fallbacks: dict[str, str] = {}


def resolve_secret(env_var: str, *, length: int = 64) -> str:
    """Return the secret for *env_var*, replacing the insecure placeholder.

    If the environment variable is unset or equals ``"super_secret"``, a
    cryptographically random URL-safe key is generated (and cached for the
    process lifetime) and a warning is logged.  Otherwise the configured
    value is returned unchanged.
    """
    value = os.getenv(env_var, "")
    if value and value != _INSECURE_PLACEHOLDER:
        return value

    if env_var not in _generated_fallbacks:
        _generated_fallbacks[env_var] = secrets.token_urlsafe(length)
        logger.warning(
            "%s is unset or still the insecure default %r; using a random "
            "per-process key. Set %s to a long random string in production "
            "(required for multi-instance deployments).",
            env_var,
            _INSECURE_PLACEHOLDER,
            env_var,
        )
    return _generated_fallbacks[env_var]
