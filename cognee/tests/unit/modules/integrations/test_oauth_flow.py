"""Unit tests for the provider-agnostic state signing in
cognee.modules.integrations.oauth_flow.

Pure functions — no DB, no network, no provider settings. Unlike
test_oauth_state.py (which tests the Slack adapter's thin wrapper and its
settings-based secret), these pass the signing secret explicitly, since this
module has no notion of "the configured provider" at all.
"""

import time
from uuid import uuid4

import pytest

from cognee.modules.integrations.oauth_flow import (
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
    # — this is exactly what lets each OAuthIntegration own its own secret.
    user_id = uuid4()
    state_a = make_state(user_id, signing_secret="secret-a")
    state_b = make_state(user_id, signing_secret="secret-b")
    assert validate_state(state_a, signing_secret="secret-b") is None
    assert validate_state(state_b, signing_secret="secret-a") is None
    assert validate_state(state_a, signing_secret="secret-a") == user_id
    assert validate_state(state_b, signing_secret="secret-b") == user_id
