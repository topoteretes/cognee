"""Slash command handling for the Slack app.

``/cognee-ask`` and ``/cognee-link`` (per-member account linking — see
handle_slack_link.py) are implemented today. ``/cognee-remember`` and
``/cognee-forget`` (async ``response_url`` pattern) are natural follow-ons
once this integration has an owner to prioritize them.

Slash command bodies are ``application/x-www-form-urlencoded`` — parsed here
from the verified raw bytes, never via FastAPI form parameters (see
verify_slack_signature's module docstring for why).

The per-channel allowlist (``provider_metadata["allowed_channel_ids"]``, set
via ``PUT /api/v1/slack/channels``) is enforced once here, before dispatch —
every future command shares this check for free instead of each handler
re-implementing it.
"""

from typing import Any
from urllib.parse import parse_qs

from cognee.modules.integrations.slack.handle_cognee_ask import handle_cognee_ask
from cognee.modules.integrations.slack.handle_slack_link import handle_cognee_link
from cognee.modules.integrations.slack.persistence import get_by_team, is_active


def _ephemeral(text: str) -> dict[str, Any]:
    # Ephemeral = visible only to the invoking user; a workspace-visible
    # reply for an unlinked/unconnected state would just be noise.
    return {"response_type": "ephemeral", "text": text}


async def handle_slack_command(raw_body: bytes) -> dict[str, Any]:
    """Answer one signature-verified slash command within Slack's 3-second window."""
    form = parse_qs(raw_body.decode())
    command = (form.get("command") or [""])[0]
    team_id = (form.get("team_id") or [""])[0]
    channel_id = (form.get("channel_id") or [""])[0]

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        return _ephemeral(
            "This Slack workspace is not connected to Cognee. "
            "Connect it from your Integrations settings."
        )

    # An empty allowlist means unrestricted — channel scoping is opt-in via
    # the Integrations page, so a workspace that never configures one keeps
    # working everywhere, exactly as before this feature existed.
    allowed_channel_ids = (credential.provider_metadata or {}).get("allowed_channel_ids") or []
    if allowed_channel_ids and channel_id not in allowed_channel_ids:
        return _ephemeral("Cognee isn't enabled in this channel.")

    if command == "/cognee-ask":
        return await handle_cognee_ask(raw_body)

    if command == "/cognee-link":
        return await handle_cognee_link(raw_body)

    return _ephemeral(f"Command `{command}` is not yet supported.")
