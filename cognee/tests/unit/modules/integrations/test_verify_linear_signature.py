"""Unit tests for cognee.modules.integrations.linear.verify_linear_signature.

No DB, no network — the request is a stub with raw bytes and headers. The
invariants that matter: only a hex HMAC-SHA256 keyed with the webhook secret
over the exact raw bytes passes; the body's ``webhookTimestamp`` replay
guard runs strictly AFTER the HMAC (a timestamp from unverified bytes proves
nothing); and a stale/missing timestamp is a 401 even with a valid HMAC.
"""

import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException

from cognee.modules.integrations.linear.verify_linear_signature import (
    LinearWebhookVerifier,
    has_fresh_webhook_timestamp,
    is_valid_linear_signature,
)

_SECRET = "It's a Secret to Everybody"


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.integrations.linear.linear_settings.linear_settings.webhook_secret",
        _SECRET,
    )


def _sign(body: bytes, secret: str = _SECRET) -> str:
    # Linear signs the raw body as bare hex — no sha256= prefix.
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _body(timestamp_ms=...) -> bytes:
    payload = {"type": "Issue", "action": "create", "organizationId": "org-1"}
    if timestamp_ms is ...:
        timestamp_ms = int(time.time() * 1000)
    if timestamp_ms is not None:
        payload["webhookTimestamp"] = timestamp_ms
    return json.dumps(payload).encode()


class _FakeRequest:
    def __init__(self, raw_body: bytes, headers: dict):
        self._raw_body = raw_body
        self.headers = headers

    async def body(self) -> bytes:
        return self._raw_body


def test_valid_signature_passes():
    body = _body()
    assert is_valid_linear_signature(body, _sign(body))


def test_tampered_body_fails():
    signature = _sign(b'{"action":"create"}')
    assert not is_valid_linear_signature(b'{"action":"remove"}', signature)


def test_wrong_secret_fails():
    body = _body()
    assert not is_valid_linear_signature(body, _sign(body, secret="other-secret"))


def test_missing_signature_fails():
    assert not is_valid_linear_signature(_body(), "")
    assert not is_valid_linear_signature(_body(), None)


def test_stale_timestamp_is_not_fresh():
    two_minutes_ago = int((time.time() - 120) * 1000)
    assert not has_fresh_webhook_timestamp(_body(two_minutes_ago))


def test_missing_or_malformed_timestamp_is_not_fresh():
    # A delivery without replay protection is unverifiable, same as stale.
    assert not has_fresh_webhook_timestamp(_body(None))
    assert not has_fresh_webhook_timestamp(b"not json")


@pytest.mark.asyncio
async def test_verify_returns_the_raw_body_on_a_valid_delivery():
    body = _body()
    request = _FakeRequest(body, {"linear-signature": _sign(body)})

    assert await LinearWebhookVerifier().verify(request) == body


@pytest.mark.asyncio
async def test_verify_rejects_a_missing_signature_header():
    request = _FakeRequest(_body(), {})

    with pytest.raises(HTTPException) as excinfo:
        await LinearWebhookVerifier().verify(request)
    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_rejects_a_stale_timestamp_even_with_a_valid_hmac():
    body = _body(int((time.time() - 120) * 1000))
    request = _FakeRequest(body, {"linear-signature": _sign(body)})

    with pytest.raises(HTTPException) as excinfo:
        await LinearWebhookVerifier().verify(request)
    assert excinfo.value.status_code == 401
    assert "timestamp" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_verify_checks_the_hmac_before_reading_the_timestamp():
    # A bad signature with a stale timestamp must report the SIGNATURE
    # failure: the timestamp lives inside the body, and parsing (let alone
    # trusting) unverified bytes would defeat the guard.
    body = _body(int((time.time() - 120) * 1000))
    request = _FakeRequest(body, {"linear-signature": _sign(body, secret="other-secret")})

    with pytest.raises(HTTPException) as excinfo:
        await LinearWebhookVerifier().verify(request)
    assert excinfo.value.status_code == 401
    assert "signature" in excinfo.value.detail.lower()
