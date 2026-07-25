"""Public Slack webhook receiver.

Three unauthenticated-by-design endpoints — Slack cannot send a bearer token
or cookie, so the ``X-Slack-Signature`` HMAC (enforced by the
``verify_slack_request`` dependency) is the entire auth model. Handlers
receive the verified raw bytes and parse them themselves; adding a parsed
body parameter here would break verification (see verify_slack_signature's
module docstring).

These are shared, multi-workspace URLs: every payload carries a ``team_id``
that the modules-layer handlers resolve back to the connecting cognee user.
"""

from fastapi import APIRouter, Depends

from cognee.modules.integrations.slack.handle_slack_command import handle_slack_command
from cognee.modules.integrations.slack.handle_slack_event import handle_slack_event
from cognee.modules.integrations.slack.handle_slack_interactive import handle_slack_interactive
from cognee.modules.integrations.slack.verify_slack_signature import verify_slack_request


def get_slack_router():
    slack_router = APIRouter()

    @slack_router.post("/commands", include_in_schema=False)
    async def slack_commands(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_command(raw_body)

    @slack_router.post("/events", include_in_schema=False)
    async def slack_events(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_event(raw_body)

    @slack_router.post("/interactive", include_in_schema=False)
    async def slack_interactive(raw_body: bytes = Depends(verify_slack_request)):
        return await handle_slack_interactive(raw_body)

    return slack_router
