"""Publish the Cognee App Home tab.

Slack shows a blank Home tab until the app calls ``views.publish`` — this is
what makes clicking Cognee in the sidebar show anything at all. The view is
static/informational: Cognee has no Slack-user <-> cognee-user linking yet
(that's CLO-240), so this can't be personalized per viewer, only explain how
to use ``/cognee-ask``.
"""

from typing import Any

import aiohttp

_VIEWS_PUBLISH_URL = "https://slack.com/api/views.publish"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _build_home_view() -> dict[str, Any]:
    return {
        "type": "home",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "Cognee", "emoji": True}},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Cognee turns your team's conversations and docs into a searchable memory.",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Ask a question*\nType `/cognee-ask <your question>` in any connected channel.",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Manage which channels can use Cognee from your Integrations settings.",
                    }
                ],
            },
        ],
    }


async def publish_home_view(access_token: str, slack_user_id: str) -> None:
    """Push the static Home tab content for one Slack user.

    Raises ``RuntimeError`` naming Slack's error code on rejection — the
    caller (the ``app_home_opened`` handler) treats this as best-effort and
    swallows it, since a broken Home tab must never fail the surrounding
    event ack. A ``not_enabled`` error here means the Home tab feature itself
    isn't turned on in the app's config (App Home -> Home Tab), not a scope
    problem — ``views.publish`` needs no OAuth scope beyond a bot token.
    """
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.post(
            _VIEWS_PUBLISH_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            json={"user_id": slack_user_id, "view": _build_home_view()},
        ) as response:
            payload = await response.json()

    if not payload.get("ok"):
        raise RuntimeError(f"Slack views.publish failed: {payload.get('error', 'unknown')}")
