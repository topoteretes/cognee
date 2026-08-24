"""GitHub webhook delivery verification.

Every inbound delivery is authenticated by an HMAC-SHA256 over the raw body
keyed with the app's webhook secret, compared against the
``X-Hub-Signature-256`` header
(https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries).

The check MUST run over the raw request bytes, before any parsing —
re-serializing a parsed JSON body changes key order/escaping and breaks the
HMAC (same rule as Slack's verifier). Unlike Slack, GitHub's scheme carries
no timestamp, so there is no replay window to enforce here; handlers are
idempotent instead (revokes are no-ops on revoked rows, syncs skip unchanged
snapshots).
"""

import hashlib
import hmac

from fastapi import HTTPException, Request

from cognee.modules.integrations.base import WebhookVerifier
from cognee.modules.integrations.github.github_settings import require

_SIGNATURE_PREFIX = "sha256="


def is_valid_github_signature(raw_body: bytes, signature: str) -> bool:
    """Whether ``signature`` is an authentic GitHub signature for ``raw_body``.

    Pure function of its inputs — the verifier and the unit tests share it.
    """
    if not signature or not signature.startswith(_SIGNATURE_PREFIX):
        return False

    expected = (
        _SIGNATURE_PREFIX
        + hmac.new(require("webhook_secret").encode(), raw_body, hashlib.sha256).hexdigest()
    )
    # compare_digest, not ==: a timing-safe comparison so the signature cannot
    # be brute-forced byte by byte from response latencies.
    return hmac.compare_digest(expected, signature)


class GithubWebhookVerifier(WebhookVerifier):
    async def verify(self, request: Request) -> bytes:
        raw_body = await request.body()
        signature = request.headers.get("x-hub-signature-256", "")

        if not is_valid_github_signature(raw_body, signature):
            raise HTTPException(status_code=401, detail="Invalid GitHub signature")

        return raw_body
