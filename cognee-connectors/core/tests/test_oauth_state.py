"""Unit tests for the provider-agnostic state signing in
cognee_connectors_core.oauth_state.

Ported from cognee/tests/unit/modules/integrations/test_oauth_flow.py and
test_oauth_state.py (cognee dev, 2026-10-26). Pure functions — no DB, no
network, no provider settings. The first half passes the signing secret
explicitly, since this module has no notion of "the configured provider" at
all. The second half replays the Slack adapter's thin wrapper — a settings-
sourced secret — against this module, which is the shape every provider's
``oauth.py`` in cognee has; the wrapper itself stayed in cognee, so only the
parts exercising oauth_state are kept.
"""

import time
from uuid import uuid4

import pytest

from cognee_connectors_core.oauth_state import (
    DEFAULT_STATE_TTL_SECONDS,
    make_state,
    sign_state_payload,
    validate_state,
)

_SECRET = "test-signing-secret"


def test_roundtrip_returns_original_user_id():
    user_id = uuid4()
    state = make_state(user_id, signing_secret=_SECRET)
    assert validate_state(state, signing_secret=_SECRET) == user_id


def test_tampered_user_id_fails():
    state = make_state(uuid4(), signing_secret=_SECRET)
    _, expires_part, signature = state.split(":")
    forged = f"{uuid4()}:{expires_part}:{signature}"
    assert validate_state(forged, signing_secret=_SECRET) is None


def test_expired_state_fails():
    user_id = uuid4()
    expires = int(time.time()) - 1
    payload = f"{user_id}:{expires}"
    state = f"{payload}:{sign_state_payload(payload, signing_secret=_SECRET)}"
    assert validate_state(state, signing_secret=_SECRET) is None


def test_custom_ttl_is_honored():
    user_id = uuid4()
    state = make_state(user_id, signing_secret=_SECRET, ttl_seconds=1)
    assert validate_state(state, signing_secret=_SECRET) == user_id
    # Force an already-expired state by minting with a negative ttl.
    expired = make_state(user_id, signing_secret=_SECRET, ttl_seconds=-1)
    assert validate_state(expired, signing_secret=_SECRET) is None


def test_default_ttl_is_ten_minutes():
    """Long enough to click through a consent screen, short enough to go stale."""
    assert DEFAULT_STATE_TTL_SECONDS == 600


def test_malformed_state_fails():
    assert validate_state("", signing_secret=_SECRET) is None
    assert validate_state(None, signing_secret=_SECRET) is None
    assert validate_state("a:b", signing_secret=_SECRET) is None
    assert validate_state("not-a-uuid:123:deadbeef", signing_secret=_SECRET) is None


def test_state_signed_with_other_secret_fails():
    state = make_state(uuid4(), signing_secret=_SECRET)
    assert validate_state(state, signing_secret="rotated-secret") is None


def test_different_signing_secrets_are_isolated():
    # Two "providers" with different secrets never validate each other's state
    # — this is exactly what lets each connector own its own secret.
    user_id = uuid4()
    state_a = make_state(user_id, signing_secret="secret-a")
    state_b = make_state(user_id, signing_secret="secret-b")
    assert validate_state(state_a, signing_secret="secret-b") is None
    assert validate_state(state_b, signing_secret="secret-a") is None
    assert validate_state(state_a, signing_secret="secret-a") == user_id
    assert validate_state(state_b, signing_secret="secret-b") == user_id


def test_make_state_requires_at_least_one_field():
    with pytest.raises(ValueError, match="at least one field"):
        make_state(signing_secret=_SECRET)


def test_multi_field_roundtrip_returns_raw_string_tuple():
    tenant_id = uuid4()
    user_id = uuid4()
    state = make_state(tenant_id, user_id, signing_secret=_SECRET)
    assert validate_state(state, signing_secret=_SECRET, field_count=2) == (
        str(tenant_id),
        str(user_id),
    )


def test_multi_field_wrong_field_count_fails():
    tenant_id = uuid4()
    user_id = uuid4()
    state = make_state(tenant_id, user_id, signing_secret=_SECRET)
    # Reading a 2-field state back with field_count=1 (or 3) must fail closed
    # rather than misparse the fields/expires/hmac boundary.
    assert validate_state(state, signing_secret=_SECRET, field_count=1) is None
    assert validate_state(state, signing_secret=_SECRET, field_count=3) is None


def test_multi_field_tampered_field_fails():
    state = make_state(uuid4(), uuid4(), signing_secret=_SECRET)
    tenant_str, _user_str, expires_part, signature = state.split(":")
    forged = f"{tenant_str}:{uuid4()}:{expires_part}:{signature}"
    assert validate_state(forged, signing_secret=_SECRET, field_count=2) is None


def test_multi_field_expired_state_fails():
    tenant_id = uuid4()
    user_id = uuid4()
    expires = int(time.time()) - 1
    payload = f"{tenant_id}:{user_id}:{expires}"
    state = f"{payload}:{sign_state_payload(payload, signing_secret=_SECRET)}"
    assert validate_state(state, signing_secret=_SECRET, field_count=2) is None


# ---- A provider wrapper over a settings-sourced secret -------------------
#
# The shape cognee's slack/oauth.py has: ``make_state(user_id)`` and
# ``validate_state(state)`` that read the secret off the provider's settings
# object at call time. The state is the callback's ONLY credential (the
# callback is unauthenticated), so the invariants are strict: only an
# untampered, unexpired state minted by us validates.


class _ProviderSettings:
    signing_secret = _SECRET


provider_settings = _ProviderSettings()


def _provider_make_state(user_id):
    return make_state(user_id, signing_secret=provider_settings.signing_secret)


def _provider_validate_state(state):
    return validate_state(state, signing_secret=provider_settings.signing_secret)


@pytest.fixture(autouse=True)
def _signing_secret(monkeypatch):
    monkeypatch.setattr(provider_settings, "signing_secret", _SECRET)


def test_wrapper_roundtrip_returns_original_user_id():
    user_id = uuid4()
    assert _provider_validate_state(_provider_make_state(user_id)) == user_id


def test_wrapper_tampered_user_id_fails():
    # Swapping the user id after signing must invalidate the HMAC — this is
    # exactly the forgery that would let an attacker bind a workspace to a
    # user they don't own.
    state = _provider_make_state(uuid4())
    _, expires_part, signature = state.split(":")
    forged = f"{uuid4()}:{expires_part}:{signature}"
    assert _provider_validate_state(forged) is None


def test_wrapper_expired_state_fails():
    user_id = uuid4()
    expires = int(time.time()) - 1
    payload = f"{user_id}:{expires}"
    state = f"{payload}:{sign_state_payload(payload, signing_secret=_SECRET)}"
    assert _provider_validate_state(state) is None


def test_wrapper_malformed_state_fails():
    assert _provider_validate_state("") is None
    assert _provider_validate_state(None) is None
    assert _provider_validate_state("a:b") is None
    assert _provider_validate_state("not-a-uuid:123:deadbeef") is None


def test_wrapper_state_signed_with_other_secret_fails(monkeypatch):
    state = _provider_make_state(uuid4())
    monkeypatch.setattr(provider_settings, "signing_secret", "rotated-secret")
    assert _provider_validate_state(state) is None
