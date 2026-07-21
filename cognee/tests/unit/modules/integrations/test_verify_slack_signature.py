"""Unit tests for cognee.modules.integrations.slack.verify_slack_signature.

Pure function — no DB, no network. The invariants that matter: only an
HMAC keyed with the signing secret over the exact raw bytes passes, and the
5-minute replay window is enforced in both directions.
"""

import hashlib
import hmac
import time

import pytest

from cognee.modules.integrations.slack.verify_slack_signature import is_valid_slack_signature

_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"


@pytest.fixture(autouse=True)
def _signing_secret(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.integrations.slack.slack_settings.slack_settings.signing_secret",
        _SECRET,
    )


def _sign(body: bytes, timestamp: str) -> str:
    basestring = f"v0:{timestamp}:".encode() + body
    return "v0=" + hmac.new(_SECRET.encode(), basestring, hashlib.sha256).hexdigest()


def test_valid_signature_passes():
    body = b"command=%2Fcognee-ask&team_id=T123"
    timestamp = str(int(time.time()))
    assert is_valid_slack_signature(body, timestamp, _sign(body, timestamp))


def test_tampered_body_fails():
    timestamp = str(int(time.time()))
    signature = _sign(b"command=%2Fcognee-ask", timestamp)
    assert not is_valid_slack_signature(b"command=%2Fcognee-forget", timestamp, signature)


def test_wrong_secret_fails(monkeypatch):
    body = b"payload"
    timestamp = str(int(time.time()))
    wrong = (
        "v0="
        + hmac.new(b"other-secret", f"v0:{timestamp}:".encode() + body, hashlib.sha256).hexdigest()
    )
    assert not is_valid_slack_signature(body, timestamp, wrong)


def test_replayed_old_timestamp_fails():
    # 6 minutes old — past the 5-minute window, even with a valid HMAC.
    body = b"payload"
    timestamp = str(int(time.time()) - 6 * 60)
    assert not is_valid_slack_signature(body, timestamp, _sign(body, timestamp))


def test_future_timestamp_fails():
    # Skew is rejected in both directions, not just the past.
    body = b"payload"
    timestamp = str(int(time.time()) + 6 * 60)
    assert not is_valid_slack_signature(body, timestamp, _sign(body, timestamp))


def test_garbage_timestamp_fails():
    assert not is_valid_slack_signature(b"payload", "not-a-number", "v0=abc")
    assert not is_valid_slack_signature(b"payload", "", "v0=abc")


def test_missing_signature_fails():
    timestamp = str(int(time.time()))
    assert not is_valid_slack_signature(b"payload", timestamp, "")
    assert not is_valid_slack_signature(b"payload", timestamp, None)
