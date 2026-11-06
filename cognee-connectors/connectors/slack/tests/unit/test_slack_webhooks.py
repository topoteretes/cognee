"""The Slack signature check on the core verifier (``cognee_connector_slack.webhooks``).

``test_verify_slack_signature.py`` (moved from cognee) covers the boolean form's
invariants; this checks that both forms are the ``slack_v0`` verifier ``CONNECTOR``
declares and agree with the core's own signing helper.
"""

import time

import pytest

from cognee_connector_slack import DESCRIPTOR
from cognee_connector_slack.webhooks import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    is_valid_slack_signature,
    slack_webhook_verifier,
    verify_slack_request_body,
)
from cognee_connectors_core.errors import WebhookVerificationError
from cognee_connectors_core.signatures import MAX_BODY_BYTES, HmacVerifier, SignatureScheme
from cognee_connectors_core.testing import sign_body

_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
_BODY = b"command=%2Fcognee-ask&team_id=T123"


@pytest.fixture(autouse=True)
def _signing_secret(monkeypatch):
    monkeypatch.setattr("cognee_connector_slack.settings.slack_settings.signing_secret", _SECRET)


def _headers(secret=_SECRET, body=_BODY, timestamp=None):
    timestamp = str(timestamp if timestamp is not None else int(time.time()))
    return {
        "X-Slack-Signature": sign_body(SignatureScheme.SLACK_V0, secret, body, timestamp=timestamp),
        "X-Slack-Request-Timestamp": timestamp,
    }


def test_verifier_is_the_connectors_slack_v0_verifier():
    verifier = slack_webhook_verifier()

    assert isinstance(verifier, HmacVerifier)
    assert verifier.provider == "slack"
    assert verifier.scheme is SignatureScheme.SLACK_V0
    assert verifier.header == SIGNATURE_HEADER == DESCRIPTOR.signature_header
    assert verifier.timestamp_header == TIMESTAMP_HEADER == DESCRIPTOR.signature_timestamp_header


@pytest.mark.asyncio
async def test_verify_slack_request_body_returns_the_raw_body_for_a_signed_request():
    # Header names in Slack's own spelling: the verifier matches them case-insensitively.
    assert await verify_slack_request_body(_BODY, _headers()) == _BODY


@pytest.mark.asyncio
async def test_verify_slack_request_body_refuses_tampering_wrong_secret_and_replay():
    for body, headers in (
        (_BODY + b"&x=1", _headers()),
        (_BODY, _headers(secret="other-secret")),
        (_BODY, _headers(timestamp=int(time.time()) - 6 * 60)),
        (_BODY, {}),
    ):
        with pytest.raises(WebhookVerificationError) as refused:
            await verify_slack_request_body(body, headers)
        assert refused.value.status_code == 401
        assert refused.value.detail == "Invalid signature"


@pytest.mark.asyncio
async def test_oversized_delivery_is_refused_with_413():
    headers = {**_headers(), "Content-Length": str(MAX_BODY_BYTES + 1)}

    with pytest.raises(WebhookVerificationError) as refused:
        await verify_slack_request_body(_BODY, headers)
    assert refused.value.status_code == 413


def test_is_valid_slack_signature_agrees_with_the_core_signing_helper():
    timestamp = str(int(time.time()))
    signature = sign_body(SignatureScheme.SLACK_V0, _SECRET, _BODY, timestamp=timestamp)

    assert is_valid_slack_signature(_BODY, timestamp, signature)
    assert not is_valid_slack_signature(_BODY + b"x", timestamp, signature)
    assert not is_valid_slack_signature(
        _BODY, timestamp, sign_body(SignatureScheme.SLACK_V0, "other", _BODY, timestamp=timestamp)
    )


def test_unconfigured_signing_secret_refuses_everything(monkeypatch):
    monkeypatch.setattr("cognee_connector_slack.settings.slack_settings.signing_secret", "")
    timestamp = str(int(time.time()))
    signature = sign_body(SignatureScheme.SLACK_V0, "", _BODY, timestamp=timestamp)

    # Never an HMAC under an empty key, even for a sender that signed with one.
    assert not is_valid_slack_signature(_BODY, timestamp, signature)


@pytest.mark.asyncio
async def test_is_valid_slack_signature_works_inside_a_running_event_loop():
    # The boolean form drives the verifier itself, so a handler running under
    # an event loop can call it without asyncio.run().
    timestamp = str(int(time.time()))
    signature = sign_body(SignatureScheme.SLACK_V0, _SECRET, _BODY, timestamp=timestamp)

    assert is_valid_slack_signature(_BODY, timestamp, signature)
