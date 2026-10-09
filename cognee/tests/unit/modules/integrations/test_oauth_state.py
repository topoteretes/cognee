"""Unit tests for the OAuth state in cognee.modules.integrations.slack.oauth.

Pure functions — no DB, no network. The state is the callback's ONLY
credential (the callback is unauthenticated), so the invariants are strict:
only an untampered, unexpired state minted by us validates.
"""

import time
from uuid import uuid4

import pytest

from cognee.modules.integrations.slack import oauth

_SECRET = "test-signing-secret"


@pytest.fixture(autouse=True)
def _signing_secret(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.integrations.slack.slack_settings.slack_settings.signing_secret",
        _SECRET,
    )


def test_roundtrip_returns_original_user_id():
    user_id = uuid4()
    assert oauth.validate_state(oauth.make_state(user_id)) == user_id


def test_tampered_user_id_fails():
    # Swapping the user id after signing must invalidate the HMAC — this is
    # exactly the forgery that would let an attacker bind a workspace to a
    # user they don't own.
    state = oauth.make_state(uuid4())
    _, expires_part, signature = state.split(":")
    forged = f"{uuid4()}:{expires_part}:{signature}"
    assert oauth.validate_state(forged) is None


def test_expired_state_fails():
    user_id = uuid4()
    expires = int(time.time()) - 1
    payload = f"{user_id}:{expires}"
    state = f"{payload}:{oauth._sign_state(payload)}"
    assert oauth.validate_state(state) is None


def test_malformed_state_fails():
    assert oauth.validate_state("") is None
    assert oauth.validate_state(None) is None
    assert oauth.validate_state("a:b") is None
    assert oauth.validate_state("not-a-uuid:123:deadbeef") is None


def test_state_signed_with_other_secret_fails(monkeypatch):
    state = oauth.make_state(uuid4())
    monkeypatch.setattr(
        "cognee.modules.integrations.slack.slack_settings.slack_settings.signing_secret",
        "rotated-secret",
    )
    assert oauth.validate_state(state) is None
