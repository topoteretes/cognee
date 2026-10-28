"""The cognee shim for Linear webhook verification.

Moved from cognee.modules.integrations.linear.verify_linear_signature (cognee dev,
2026-10-26): the verification itself now lives in :mod:`cognee_connector_linear.webhooks`
on the core contract (the HMAC over the raw bytes, then the body's ``webhookTimestamp``
replay guard). This adapts FastAPI's ``Request`` — its body plus its headers — to that
verifier and maps a refusal to the ``HTTPException`` cognee's generic events route
answers with. The pure predicates are re-exported for callers of the former module.
"""

from cognee.modules.integrations.base import WebhookVerifier
from fastapi import HTTPException, Request

from cognee_connector_linear import webhooks
from cognee_connector_linear.webhooks import has_fresh_webhook_timestamp, is_valid_linear_signature
from cognee_connectors_core.errors import WebhookVerificationError


class LinearWebhookVerifier(WebhookVerifier):
    """cognee's request-based verifier, delegating to the contract's."""

    def __init__(self, verifier: webhooks.LinearWebhookVerifier | None = None) -> None:
        self._verifier = verifier if verifier is not None else webhooks.LinearWebhookVerifier()

    async def verify(self, request: Request) -> bytes:
        raw_body = await request.body()
        try:
            return await self._verifier.verify(raw_body, request.headers)
        except WebhookVerificationError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from None


__all__ = ["LinearWebhookVerifier", "has_fresh_webhook_timestamp", "is_valid_linear_signature"]
