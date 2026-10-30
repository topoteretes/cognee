"""The FastAPI dependency that verifies a Slack request on a cognee server.

Moved from cognee.modules.integrations.slack.verify_slack_signature (cognee dev,
2026-10-27). The HMAC check itself moved to :mod:`cognee_connector_slack.webhooks`,
on the core verifier; this is the shim that adapts FastAPI's ``Request`` (raw body
plus headers) to it and maps a refusal to the ``HTTPException`` the routes have
always answered with. ``is_valid_slack_signature`` is re-exported for callers of
the old module.
"""

from fastapi import HTTPException, Request

from cognee_connector_slack.webhooks import is_valid_slack_signature, verify_slack_request_body
from cognee_connectors_core.errors import WebhookVerificationError

__all__ = ["is_valid_slack_signature", "verify_slack_request"]


async def verify_slack_request(request: Request) -> bytes:
    """FastAPI dependency: verify the Slack signature and return the raw body."""
    raw_body = await request.body()
    try:
        return await verify_slack_request_body(raw_body, request.headers)
    except WebhookVerificationError as error:
        if error.status_code == 401:
            # The wording the Slack routes always answered a bad signature with.
            raise HTTPException(status_code=401, detail="Invalid Slack signature") from None
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None
