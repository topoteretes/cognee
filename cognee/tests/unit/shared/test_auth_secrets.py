"""Tests for cognee.shared.auth_secrets (#4265)."""

import os

import pytest

from cognee.shared import auth_secrets
from cognee.shared.auth_secrets import resolve_secret


@pytest.fixture(autouse=True)
def _clear_fallback_cache():
    """Ensure each test starts with an empty generated-fallback cache."""
    auth_secrets._generated_fallbacks.clear()
    yield
    auth_secrets._generated_fallbacks.clear()


def test_configured_secret_returned_unchanged(monkeypatch):
    """A real configured secret must be returned verbatim."""
    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "my-real-production-secret")
    assert resolve_secret("FASTAPI_USERS_JWT_SECRET") == "my-real-production-secret"


def test_insecure_placeholder_replaced_with_random(monkeypatch):
    """The well-known insecure placeholder must NOT be returned as-is."""
    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "super_secret")
    secret = resolve_secret("FASTAPI_USERS_JWT_SECRET")
    assert secret != "super_secret"
    assert len(secret) >= 32  # token_urlsafe(64) yields a long key


def test_unset_env_replaced_with_random(monkeypatch):
    """An unset secret must also be replaced with a random key."""
    monkeypatch.delenv("FASTAPI_USERS_JWT_SECRET", raising=False)
    secret = resolve_secret("FASTAPI_USERS_JWT_SECRET")
    assert secret != "super_secret"
    assert len(secret) >= 32


def test_fallback_is_stable_within_process(monkeypatch):
    """Repeated lookups for the same env var return the same generated key,
    so tokens signed earlier remain verifiable within the process."""
    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "super_secret")
    first = resolve_secret("FASTAPI_USERS_JWT_SECRET")
    second = resolve_secret("FASTAPI_USERS_JWT_SECRET")
    assert first == second


def test_different_env_vars_get_different_keys(monkeypatch):
    """Distinct env vars must not share the same generated fallback."""
    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "super_secret")
    monkeypatch.setenv("FASTAPI_USERS_VERIFICATION_TOKEN_SECRET", "super_secret")
    jwt_key = resolve_secret("FASTAPI_USERS_JWT_SECRET")
    verification_key = resolve_secret("FASTAPI_USERS_VERIFICATION_TOKEN_SECRET")
    assert jwt_key != verification_key


def test_get_token_module_does_not_use_placeholder(monkeypatch):
    """Regression: cognee.get_token.SECRET_KEY must never be 'super_secret'."""
    monkeypatch.setenv("FASTAPI_USERS_JWT_SECRET", "super_secret")
    auth_secrets._generated_fallbacks.clear()

    import importlib

    from cognee import get_token

    importlib.reload(get_token)
    assert get_token.SECRET_KEY != "super_secret"
