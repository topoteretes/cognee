"""Events API dispatch for the Slack app.

Handles:

* ``app_uninstalled`` / ``tokens_revoked`` — lifecycle hygiene, revokes the
  stored installation so no dangling bot token survives an uninstall.
* ``app_home_opened`` — publishes the (static, informational) Home tab
  content. Slack shows a blank Home tab until the app calls
  ``views.publish``, so this is what makes clicking Cognee in the sidebar
  show anything at all.

Slack delivers events at-least-once (3 retries: immediate, 1 min, 5 min) and
does NOT guarantee ordering between event types, so every handler here is
idempotent and independent of the others.

Message/mention events for ingestion are deliberately absent — that requires
Slack Marketplace approval (non-Marketplace apps get channel-history reads
throttled to 1 req/min).
"""

import json
import logging
from typing import Any, Optional

from cognee.modules.integrations.credentials import decrypt_token_payload
from cognee.modules.integrations.slack.home import publish_home_view
from cognee.modules.integrations.slack.persistence import get_by_team, is_active, revoke_by_team

logger = logging.getLogger(__name__)

_REVOKING_EVENT_TYPES = {"app_uninstalled", "tokens_revoked"}


async def handle_slack_event(raw_body: bytes) -> dict[str, Any]:
    """Dispatch one signature-verified Events API request body.

    Returns the JSON-serializable response for the router. Always answers
    within the 3-second window — ``app_home_opened``'s actual work (one
    ``views.publish`` call) comfortably fits that budget, and never turns
    into a non-200 response regardless; see ``_publish_home_view``.
    """
    envelope = json.loads(raw_body)

    # One-time endpoint ownership handshake: Slack POSTs a challenge when the
    # request URL is first configured and expects it echoed back.
    if envelope.get("type") == "url_verification":
        return {"challenge": envelope.get("challenge", "")}

    if envelope.get("type") != "event_callback":
        return {"ok": True}

    event = envelope.get("event") or {}
    event_type = event.get("type")
    team_id = envelope.get("team_id")

    if event_type in _REVOKING_EVENT_TYPES and team_id:
        found = await revoke_by_team(team_id)
        logger.info(
            "Slack %s for team %s — installation %s",
            event_type,
            team_id,
            "revoked" if found else "not found",
        )
    elif event_type == "app_home_opened" and team_id:
        await _publish_home_view(team_id, event.get("user"))

    # Unhandled event types are acked, not errored: erroring makes Slack
    # retry (and eventually disable the endpoint), which can never improve
    # the outcome for an event this app doesn't act on.
    return {"ok": True}


async def _publish_home_view(team_id: str, slack_user_id: Optional[str]) -> None:
    """Best-effort Home tab refresh — never raises.

    A broken Home tab is a cosmetic problem, not a reason to fail this
    event's ack — that would make Slack retry three times over six minutes
    for no benefit, since a views.publish failure won't fix itself on retry
    any faster than on the next app_home_opened.
    """
    if not slack_user_id:
        return

    credential = await get_by_team(team_id)
    if not is_active(credential):
        return

    try:
        access_token = decrypt_token_payload(credential).get("access_token")
        if access_token:
            await publish_home_view(access_token, slack_user_id)
    except Exception:  # noqa: BLE001 - a broken Home tab must never fail the event ack
        logger.exception("Failed to publish App Home view for team %s", team_id)
