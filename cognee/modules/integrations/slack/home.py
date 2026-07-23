"""Publish the Cognee App Home tab.

Slack shows a blank Home tab until the app calls ``views.publish`` — this is
what makes clicking Cognee in the sidebar show anything at all. The view is
static/informational (same content for every viewer) — it exists to answer
the "what do I even do with this app, and what does it need from me?"
question, since a new member has no other way to discover ``/cognee-link``
before ever needing it (there is nothing prompting them to run it until
they try ``/cognee-ask`` and get turned away — see handle_cognee_ask.py).
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
                    "text": (
                        "*1. Link your Cognee account*\n"
                        "Every workspace member needs their own link before "
                        '`/cognee-ask` or "Remember this" will use *their* '
                        "memory instead of just the person who installed this app.\n"
                        "Run `/cognee-link` — it replies privately with a link. Open it "
                        "in a browser where you're already logged in to Cognee and "
                        "click *Confirm*. Nothing to copy or paste."
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*2. Ask a question*\n"
                        "Type `/cognee-ask <your question>` in any connected "
                        "channel. Nothing is posted publicly until you review "
                        "the answer and choose *Share* — *Discard* drops it, "
                        "visible to no one."
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*3. Remember something*\n"
                        "Hover any message → *More actions* → *Connect to "
                        "apps* → *Remember this* to save it to your memory."
                    ),
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
