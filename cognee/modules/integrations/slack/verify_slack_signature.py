"""Slack request signature verification.

Every inbound Slack payload — slash commands, Events API, interactivity — is
authenticated by an HMAC over ``v0:{timestamp}:{raw_body}`` keyed with the
app's signing secret, compared against the ``X-Slack-Signature`` header
(https://docs.slack.dev/authentication/verifying-requests-from-slack).

The check MUST run over the raw request bytes, before any parsing: slash
commands arrive form-encoded, and re-serializing a parsed dict changes key
order/escaping and breaks the HMAC. Handlers therefore take the verified raw
body from the :func:`verify_slack_request` dependency and parse it themselves
instead of declaring parsed body parameters.
"""

import hashlib
import hmac
import time

from fastapi import HTTPException, Request

from cognee.modules.integrations.slack.slack_settings import require

# Slack's documented replay-protection window: reject requests whose
# timestamp differs from local time by more than 5 minutes, so a captured
# request cannot be replayed after the fact.
_MAX_TIMESTAMP_SKEW_SECONDS = 60 * 5

_SIGNATURE_VERSION = "v0"


def is_valid_slack_signature(raw_body: bytes, timestamp: str, signature: str) -> bool:
    """Whether ``signature`` is a current, authentic Slack signature for ``raw_body``.

    Pure function of its inputs (plus the clock) — the router dependency and
    the unit tests share it.
    """
    try:
        request_time = int(timestamp)
    except (TypeError, ValueError):
        return False

    if abs(time.time() - request_time) > _MAX_TIMESTAMP_SKEW_SECONDS:
        return False

    basestring = f"{_SIGNATURE_VERSION}:{timestamp}:".encode() + raw_body
    expected = (
        _SIGNATURE_VERSION
        + "="
        + hmac.new(require("signing_secret").encode(), basestring, hashlib.sha256).hexdigest()
    )
    # compare_digest, not ==: a timing-safe comparison so the signature cannot
    # be brute-forced byte by byte from response latencies.
    return hmac.compare_digest(expected, signature or "")


async def verify_slack_request(request: Request) -> bytes:
    """FastAPI dependency: verify the Slack signature and return the raw body."""
    raw_body = await request.body()
    timestamp = request.headers.get("x-slack-request-timestamp", "")
    signature = request.headers.get("x-slack-signature", "")

    if not is_valid_slack_signature(raw_body, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    return raw_body
