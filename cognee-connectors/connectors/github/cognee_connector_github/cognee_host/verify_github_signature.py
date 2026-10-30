"""GitHub webhook verification on cognee's FastAPI-bound ``WebhookVerifier``.

Moved from cognee.modules.integrations.github.verify_github_signature (cognee
dev, 2026-10-26). The HMAC check itself now lives in
:mod:`cognee_connector_github.webhooks` on the core verifier; this shim adapts
FastAPI's ``Request`` (body plus headers) to it and maps a refused delivery to
the ``HTTPException`` cognee's generic events route answers with.
"""

from cognee.modules.integrations.base import WebhookVerifier
from fastapi import HTTPException, Request

from cognee_connector_github.webhooks import github_webhook_verifier
from cognee_connectors_core.errors import WebhookVerificationError


class GithubWebhookVerifier(WebhookVerifier):
    async def verify(self, request: Request) -> bytes:
        raw_body = await request.body()
        try:
            return await github_webhook_verifier().verify(raw_body, request.headers)
        except WebhookVerificationError as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error


__all__ = ["GithubWebhookVerifier"]
