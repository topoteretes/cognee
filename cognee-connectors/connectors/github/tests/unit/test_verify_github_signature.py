"""Unit tests for cognee_connector_github.webhooks.github_webhook_verifier.

No DB, no network. The invariant that matters: only an
HMAC-SHA256 keyed with the webhook secret over the exact raw bytes, carried
in GitHub's ``sha256=`` header format, passes.
"""

import hashlib
import hmac

import pytest

from cognee_connector_github.webhooks import SIGNATURE_HEADER, github_webhook_verifier
from cognee_connectors_core.errors import WebhookVerificationError

_SECRET = "It's a Secret to Everybody"


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    monkeypatch.setattr(
        "cognee_connector_github.settings.github_settings.webhook_secret",
        _SECRET,
    )


async def _accepted(body: bytes, signature: str | None) -> bool:
    """Whether the verifier the route runs accepts this delivery."""
    headers = {} if signature is None else {SIGNATURE_HEADER: signature}
    try:
        return await github_webhook_verifier().verify(body, headers) == body
    except WebhookVerificationError:
        return False


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_valid_signature_passes():
    body = b'{"action":"created","installation":{"id":42}}'
    assert await _accepted(body, _sign(body))


@pytest.mark.asyncio
async def test_tampered_body_fails():
    signature = _sign(b'{"action":"created"}')
    assert not await _accepted(b'{"action":"deleted"}', signature)


@pytest.mark.asyncio
async def test_wrong_secret_fails():
    body = b"payload"
    wrong = "sha256=" + hmac.new(b"other-secret", body, hashlib.sha256).hexdigest()
    assert not await _accepted(body, wrong)


@pytest.mark.asyncio
async def test_missing_prefix_fails():
    body = b"payload"
    bare_hex = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert not await _accepted(body, bare_hex)
    # sha1= is GitHub's legacy header, deliberately unsupported.
    assert not await _accepted(body, "sha1=" + bare_hex)


@pytest.mark.asyncio
async def test_missing_signature_fails():
    assert not await _accepted(b"payload", "")
    assert not await _accepted(b"payload", None)


@pytest.mark.asyncio
async def test_non_ascii_signature_fails():
    assert not await _accepted(b"payload", "sha256=\xff")


@pytest.mark.asyncio
async def test_unset_secret_refuses_the_delivery(monkeypatch):
    monkeypatch.setattr("cognee_connector_github.settings.github_settings.webhook_secret", None)
    assert not await _accepted(b"payload", _sign(b"payload"))
