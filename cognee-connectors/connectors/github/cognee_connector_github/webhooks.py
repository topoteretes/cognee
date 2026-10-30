"""GitHub webhook delivery verification.

Moved from cognee.modules.integrations.github.verify_github_signature (cognee
dev, 2026-10-26) onto the core contract: the check itself is the core
``HmacVerifier`` with GitHub's ``sha256_prefix`` scheme, and the FastAPI shim
that adapts a request to it stays in ``cognee_host/verify_github_signature.py``.

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
from typing import Any

from cognee_connector_github.settings import github_settings, require
from cognee_connectors_core.signatures import HmacVerifier, SignatureScheme

PROVIDER = "github"

#: The header GitHub carries the delivery signature in.
SIGNATURE_HEADER = "x-hub-signature-256"

_SIGNATURE_PREFIX = "sha256="


def is_valid_github_signature(raw_body: bytes, signature: str) -> bool:
    """Whether ``signature`` is an authentic GitHub signature for ``raw_body``.

    Pure function of its inputs — the unit tests share it with the verifier
    below as the reference computation.
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


def github_webhook_verifier(settings: Any = None) -> HmacVerifier:
    """GitHub's ``X-Hub-Signature-256`` on the core verifier.

    ``settings`` is any object with a ``webhook_secret`` attribute (the
    package's :data:`~cognee_connector_github.settings.github_settings` by
    default). The secret is read per request, so a rotated secret takes
    effect on the next delivery; an unset secret makes the verifier refuse
    every delivery rather than check against an empty key.
    """
    source = github_settings if settings is None else settings
    return HmacVerifier(
        provider=PROVIDER,
        scheme=SignatureScheme.SHA256_PREFIX,
        header=SIGNATURE_HEADER,
        secret_provider=lambda: getattr(source, "webhook_secret", None) or None,
    )


__all__ = ["PROVIDER", "SIGNATURE_HEADER", "github_webhook_verifier", "is_valid_github_signature"]
