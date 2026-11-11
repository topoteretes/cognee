"""Unit tests for the OAuth state every integration's install flow carries.

Pure functions, no DB, no network. The state is the callback's ONLY credential (the
callback is unauthenticated), so the invariants are strict: only an untampered,
unexpired state minted with this server's own key validates.
"""

import base64
import time
from uuid import uuid4

import pytest

from cognee.modules.integrations.crypto import oauth_state_signing_key
from cognee.modules.integrations.oauth_flow import (
    make_state,
    sign_state_payload,
    validate_state,
)


@pytest.fixture(autouse=True)
def _credentials_key(monkeypatch):
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEYS", raising=False)
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID", raising=False)
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"k" * 32).decode())


def test_roundtrip_returns_original_user_id():
    user_id = uuid4()
    key = oauth_state_signing_key()
    assert validate_state(make_state(user_id, signing_secret=key), signing_secret=key) == user_id


def test_tampered_user_id_fails():
    # Swapping the user id after signing must invalidate the HMAC: this is exactly the
    # forgery that would let an attacker bind an account to a user they don't own.
    key = oauth_state_signing_key()
    _, expires_part, signature = make_state(uuid4(), signing_secret=key).split(":")
    forged = f"{uuid4()}:{expires_part}:{signature}"
    assert validate_state(forged, signing_secret=key) is None


def test_expired_state_fails():
    key = oauth_state_signing_key()
    payload = f"{uuid4()}:{int(time.time()) - 1}"
    state = f"{payload}:{sign_state_payload(payload, signing_secret=key)}"
    assert validate_state(state, signing_secret=key) is None


def test_malformed_state_fails():
    key = oauth_state_signing_key()
    assert validate_state("", signing_secret=key) is None
    assert validate_state(None, signing_secret=key) is None  # ty: ignore[invalid-argument-type]
    assert validate_state("a:b", signing_secret=key) is None
    assert validate_state("not-a-uuid:123:deadbeef", signing_secret=key) is None


def test_the_key_is_not_the_credentials_key_itself_and_follows_its_rotation(monkeypatch):
    """The state key is derived, so a state never exposes material that decrypts tokens,
    and a rotated credentials key ends the states signed under the old one."""
    key = oauth_state_signing_key()
    assert key != (b"k" * 32).hex()
    state = make_state(uuid4(), signing_secret=key)

    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"r" * 32).decode())

    assert validate_state(state, signing_secret=oauth_state_signing_key()) is None


def test_no_credentials_key_means_no_state(monkeypatch):
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEY")
    with pytest.raises(RuntimeError, match="INTEGRATION_CREDENTIALS_KEYS"):
        oauth_state_signing_key()
