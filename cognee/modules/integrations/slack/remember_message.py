"""Remember a Slack message shortcut's content in Cognee.

Wraps :func:`cognee.remember` with a Slack-specific dataset and
provenance-carrying text — bare message text alone loses who said it and
where, both useful context for the resulting knowledge graph.
"""

from typing import Optional
from uuid import UUID

from cognee.api.v1.remember.remember import remember as cognee_remember
from cognee.modules.users.methods import get_user

SLACK_DATASET_NAME = "slack"


def _format_remembered_text(
    text: str, *, channel_name: Optional[str], author_id: Optional[str]
) -> str:
    channel = f"#{channel_name}" if channel_name else "Slack"
    author = f"<@{author_id}>" if author_id else "someone"
    return f"In {channel}, {author} said: {text}"


async def remember_message(
    user_id: UUID,
    *,
    text: str,
    channel_name: Optional[str] = None,
    author_id: Optional[str] = None,
) -> None:
    """Store one Slack message as a Cognee memory, attributed to ``user_id``.

    ``run_in_background=True`` is enough to keep this call fast (dataset
    resolution only, no LLM work before it returns) — no ``response_url``
    dance is needed the way ``/cognee-ask``'s search does, since there's no
    slow work to hide from Slack's 3-second window in the first place.
    """
    owner = await get_user(user_id)
    enriched_text = _format_remembered_text(text, channel_name=channel_name, author_id=author_id)
    await cognee_remember(
        enriched_text,
        dataset_name=SLACK_DATASET_NAME,
        user=owner,
        run_in_background=True,
    )
