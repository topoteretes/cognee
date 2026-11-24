"""GitHub webhook delivery verification.

Copied from cognee.modules.integrations.github.verify_github_signature (cognee
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

from typing import Any

from cognee_connector_github.settings import github_settings
from cognee_connectors_core.signatures import HmacVerifier, SignatureScheme

PROVIDER = "github"

#: The header GitHub carries the delivery signature in.
SIGNATURE_HEADER = "x-hub-signature-256"


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


__all__ = ["PROVIDER", "SIGNATURE_HEADER", "github_webhook_verifier"]
