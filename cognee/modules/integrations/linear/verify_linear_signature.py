"""Linear webhook delivery verification.

Every inbound delivery is authenticated by a hex HMAC-SHA256 over the raw
body keyed with the app's webhook signing secret, compared against the
``Linear-Signature`` header
(https://linear.app/developers/webhooks#securing-webhooks).

The check MUST run over the raw request bytes, before any parsing —
re-serializing a parsed JSON body changes key order/escaping and breaks the
HMAC (same rule as the Slack and GitHub verifiers). Unlike GitHub, Linear
does carry replay protection: the payload's ``webhookTimestamp`` field (Unix
milliseconds) is checked against a one-minute skew window, Linear's
documented guidance. That field lives *inside* the body, so it is read only
after the HMAC has authenticated the bytes — a timestamp from an unverified
body proves nothing.
"""

import hashlib
import hmac
import json
import time

from fastapi import HTTPException, Request

from cognee.modules.integrations.base import WebhookVerifier
from cognee.modules.integrations.linear.linear_settings import require

# Linear's documented replay-protection window: reject deliveries whose
# webhookTimestamp differs from local time by more than a minute, so a
# captured request cannot be replayed after the fact.
_MAX_TIMESTAMP_SKEW_SECONDS = 60


def is_valid_linear_signature(raw_body: bytes, signature: str) -> bool:
    """Whether ``signature`` is an authentic Linear signature for ``raw_body``.

    Pure function of its inputs — the verifier and the unit tests share it.
    """
    if not signature:
        return False

    expected = hmac.new(require("webhook_secret").encode(), raw_body, hashlib.sha256).hexdigest()
    # compare_digest, not ==: a timing-safe comparison so the signature cannot
    # be brute-forced byte by byte from response latencies.
    return hmac.compare_digest(expected, signature)


def has_fresh_webhook_timestamp(raw_body: bytes) -> bool:
    """Whether ``raw_body`` carries a ``webhookTimestamp`` within the skew window.

    Call only on a body whose HMAC already passed — parsing (and trusting a
    timestamp from) unverified bytes would defeat the point of the guard.
    A missing or malformed timestamp is rejected the same as a stale one.
    """
    try:
        payload = json.loads(raw_body)
        timestamp_ms = int(payload["webhookTimestamp"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

    return abs(time.time() - timestamp_ms / 1000) <= _MAX_TIMESTAMP_SKEW_SECONDS


class LinearWebhookVerifier(WebhookVerifier):
    async def verify(self, request: Request) -> bytes:
        raw_body = await request.body()
        signature = request.headers.get("linear-signature", "")

        if not is_valid_linear_signature(raw_body, signature):
            raise HTTPException(status_code=401, detail="Invalid Linear signature")

        # Only after the HMAC passed — see module docstring.
        if not has_fresh_webhook_timestamp(raw_body):
            raise HTTPException(status_code=401, detail="Stale Linear webhook timestamp")

        return raw_body
