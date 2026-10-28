"""Linear webhook verification on the core contract (cognee_connector_linear.webhooks).

No host, no network: the delivery is raw bytes and a headers mapping. The invariants that
matter: only a hex HMAC-SHA256 keyed with the webhook secret over the exact raw bytes passes;
the body's ``webhookTimestamp`` replay guard runs strictly AFTER the HMAC; a stale or missing
timestamp is a 401 even with a valid HMAC; and the core's body cap applies.
"""

import hashlib
import hmac
import json
import time

import pytest

from cognee_connector_linear.connector import DESCRIPTOR
from cognee_connector_linear.settings import linear_settings
from cognee_connector_linear.webhooks import (
    LinearWebhookVerifier,
    configured_webhook_secret,
    has_fresh_webhook_timestamp,
    is_valid_linear_signature,
)
from cognee_connectors_core.errors import WebhookVerificationError
from cognee_connectors_core.signatures import MAX_BODY_BYTES
from cognee_connectors_core.testing import webhook_headers

_SECRET = "It's a Secret to Everybody"


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    monkeypatch.setattr(linear_settings, "webhook_secret", _SECRET)


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


async def _verify(body, headers, verifier=None):
    return await (verifier or LinearWebhookVerifier()).verify(body, headers)


@pytest.mark.asyncio
async def test_a_valid_delivery_returns_the_raw_body():
    body = _body()

    assert await _verify(body, {"linear-signature": _sign(body)}) == body


@pytest.mark.asyncio
async def test_the_descriptors_own_headers_are_what_the_verifier_accepts():
    """The descriptor and the verifier agree on the scheme and the header name."""
    body = _body()

    headers = webhook_headers(DESCRIPTOR, _SECRET, body)

    assert list(headers) == ["linear-signature"]
    assert await _verify(body, headers) == body


@pytest.mark.asyncio
async def test_header_names_match_in_any_case():
    body = _body()

    assert await _verify(body, {"Linear-Signature": _sign(body)}) == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [{}, {"linear-signature": ""}, {"linear-signature": "sha256=" + "0" * 64}],
    ids=["missing", "empty", "prefixed"],
)
async def test_a_missing_or_malformed_signature_is_refused(headers):
    with pytest.raises(WebhookVerificationError) as caught:
        await _verify(_body(), headers)

    assert caught.value.status_code == 401
    assert caught.value.detail == "Invalid signature"


@pytest.mark.asyncio
async def test_a_tampered_body_or_a_wrong_secret_is_refused():
    body = _body()

    with pytest.raises(WebhookVerificationError):
        await _verify(body + b" ", {"linear-signature": _sign(body)})
    with pytest.raises(WebhookVerificationError):
        await _verify(body, {"linear-signature": _sign(body, secret="other-secret")})


@pytest.mark.asyncio
async def test_a_stale_timestamp_is_refused_even_with_a_valid_hmac():
    body = _body(int((time.time() - 120) * 1000))

    with pytest.raises(WebhookVerificationError) as caught:
        await _verify(body, {"linear-signature": _sign(body)})

    assert caught.value.status_code == 401
    assert "timestamp" in caught.value.detail.lower()


@pytest.mark.asyncio
async def test_a_missing_timestamp_is_refused_like_a_stale_one():
    body = _body(None)

    with pytest.raises(WebhookVerificationError) as caught:
        await _verify(body, {"linear-signature": _sign(body)})

    assert "timestamp" in caught.value.detail.lower()


@pytest.mark.asyncio
async def test_the_hmac_is_checked_before_the_timestamp_is_read():
    # A bad signature with a stale timestamp must report the SIGNATURE failure:
    # the timestamp lives inside the body, and parsing (let alone trusting)
    # unverified bytes would defeat the guard.
    body = _body(int((time.time() - 120) * 1000))

    with pytest.raises(WebhookVerificationError) as caught:
        await _verify(body, {"linear-signature": _sign(body, secret="other-secret")})

    assert caught.value.detail == "Invalid signature"


@pytest.mark.asyncio
async def test_an_unconfigured_secret_refuses_every_delivery(monkeypatch):
    monkeypatch.setattr(linear_settings, "webhook_secret", "")
    body = _body()

    assert configured_webhook_secret() is None
    with pytest.raises(WebhookVerificationError) as caught:
        await _verify(body, {"linear-signature": _sign(body, secret="")})
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_a_secret_provider_is_read_per_delivery():
    secrets = {"value": "first"}
    verifier = LinearWebhookVerifier(secret_provider=lambda: secrets["value"])
    body = _body()

    assert await _verify(body, {"linear-signature": _sign(body, "first")}, verifier) == body
    secrets["value"] = "second"
    with pytest.raises(WebhookVerificationError):
        await _verify(body, {"linear-signature": _sign(body, "first")}, verifier)
    assert await _verify(body, {"linear-signature": _sign(body, "second")}, verifier) == body


@pytest.mark.asyncio
async def test_a_body_over_the_cap_is_refused_before_the_hmac():
    body = b"{" + b" " * MAX_BODY_BYTES + b"}"

    with pytest.raises(WebhookVerificationError) as caught:
        await _verify(body, {"linear-signature": _sign(body)})

    assert caught.value.status_code == 413


def test_the_pure_predicates_agree_with_the_verifier():
    body = _body()
    assert is_valid_linear_signature(body, _sign(body))
    assert not is_valid_linear_signature(body, _sign(body, secret="other-secret"))
    assert not is_valid_linear_signature(body, "")
    assert has_fresh_webhook_timestamp(body)
    assert not has_fresh_webhook_timestamp(_body(int((time.time() - 120) * 1000)))
    assert not has_fresh_webhook_timestamp(b"not json")
