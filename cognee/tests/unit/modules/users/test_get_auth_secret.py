"""Tests for ``get_auth_secret``.

Every auth secret used to fall back to the literal ``"super_secret"`` when its
environment variable was unset, so the signing key of any default cognee
deployment was public knowledge and a forged token was accepted everywhere.
These tests hold the replacement to three properties: a configured value is
used verbatim, an unconfigured one is random rather than a shared constant, and
it stays stable within the process so a token can be verified after it is
issued.
"""

import os
from unittest.mock import patch

from cognee.modules.users.authentication.get_auth_secret import (
    JWT_SECRET_ENV_VAR,
    RESET_PASSWORD_TOKEN_SECRET_ENV_VAR,
    VERIFICATION_TOKEN_SECRET_ENV_VAR,
    get_auth_secret,
)

AUTH_SECRET_ENV_VARS = [
    JWT_SECRET_ENV_VAR,
    RESET_PASSWORD_TOKEN_SECRET_ENV_VAR,
    VERIFICATION_TOKEN_SECRET_ENV_VAR,
]


def _without_env():
    """An environment with none of the auth secrets set."""
    environment = {key: value for key, value in os.environ.items()}
    for env_var in AUTH_SECRET_ENV_VARS:
        environment.pop(env_var, None)
    return patch.dict(os.environ, environment, clear=True)


def test_configured_secret_is_used_verbatim():
    get_auth_secret.cache_clear()
    with patch.dict(os.environ, {JWT_SECRET_ENV_VAR: "a-long-random-string"}):
        assert get_auth_secret(JWT_SECRET_ENV_VAR) == "a-long-random-string"
    get_auth_secret.cache_clear()


def test_unset_secret_is_not_a_shared_constant():
    """The case that was wrong: an unset variable produced the same key everywhere."""
    get_auth_secret.cache_clear()
    with _without_env():
        secret = get_auth_secret(JWT_SECRET_ENV_VAR)

    assert secret != "super_secret"
    assert len(secret) >= 32
    get_auth_secret.cache_clear()


def test_every_auth_secret_variable_is_covered():
    """The reported issue named the JWT secret; reset and verification share the defect."""
    for env_var in AUTH_SECRET_ENV_VARS:
        get_auth_secret.cache_clear()
        with _without_env():
            assert get_auth_secret(env_var) != "super_secret"
    get_auth_secret.cache_clear()


def test_generated_secret_is_stable_within_the_process():
    """A token signed on one call must still verify on the next."""
    get_auth_secret.cache_clear()
    with _without_env():
        assert get_auth_secret(JWT_SECRET_ENV_VAR) == get_auth_secret(JWT_SECRET_ENV_VAR)
    get_auth_secret.cache_clear()


def test_each_variable_gets_its_own_generated_secret():
    """A reset-password token must not be accepted as a session token."""
    get_auth_secret.cache_clear()
    with _without_env():
        secrets_by_var = {env_var: get_auth_secret(env_var) for env_var in AUTH_SECRET_ENV_VARS}

    assert len(set(secrets_by_var.values())) == len(AUTH_SECRET_ENV_VARS)
    get_auth_secret.cache_clear()


def test_empty_secret_is_treated_as_unset():
    """An exported-but-empty variable is not a usable signing key."""
    get_auth_secret.cache_clear()
    with patch.dict(os.environ, {JWT_SECRET_ENV_VAR: ""}):
        assert get_auth_secret(JWT_SECRET_ENV_VAR) != ""
    get_auth_secret.cache_clear()


def test_every_call_site_uses_one_call_form_per_variable():
    """``lru_cache`` keys on the call signature, so a default argument would split one
    variable across two cache entries and mint two different secrets for it. Every
    caller must therefore name the variable, and ``get_auth_secret`` must not offer a
    default that lets a call site omit it.
    """
    from inspect import signature

    env_var_parameter = signature(get_auth_secret.__wrapped__).parameters["env_var"]

    assert env_var_parameter.default is env_var_parameter.empty


def test_user_manager_resolves_its_secrets_on_use():
    """``import cognee`` loads UserManager, so the secrets must not resolve at import.

    Reading them off the class rather than at class-definition time keeps a plain
    SDK import free of warnings about tokens it never mints, and lets a variable
    exported after import still take effect.
    """
    from cognee.modules.users.get_user_manager import UserManager

    manager = UserManager.__new__(UserManager)

    get_auth_secret.cache_clear()
    with patch.dict(
        os.environ,
        {
            RESET_PASSWORD_TOKEN_SECRET_ENV_VAR: "reset-secret",
            VERIFICATION_TOKEN_SECRET_ENV_VAR: "verification-secret",
        },
    ):
        assert manager.reset_password_token_secret == "reset-secret"
        assert manager.verification_token_secret == "verification-secret"
    get_auth_secret.cache_clear()
