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

# Structured origin stamp: every integration that ingests data should tag it
# with a node_set naming the source system, so the origin survives into the
# graph (NodeSet node + belongs_to_set edges + source_node_set property)
# instead of living only in the dataset name and the prose of the text.
SLACK_NODE_SET = ["slack"]


def _format_remembered_text(
    text: str, *, channel_name: Optional[str], author_id: Optional[str]
) -> str:
    channel = f"#{channel_name}" if channel_name else "Slack"
    author = f"<@{author_id}>" if author_id else "someone"
    return f"In {channel}, {author} said: {text}"


def _format_noted_text(text: str, *, author_id: Optional[str]) -> str:
    """Provenance for a note nobody said in Slack — see :func:`remember_note`.

    Deliberately "noted", not "said": the shortcut path quotes a real
    message, this one records something that happened elsewhere, and
    flattening the two would put words in someone's mouth in the graph.
    """
    author = f"<@{author_id}>" if author_id else "someone"
    return f"In Slack, {author} noted: {text}"


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
        node_set=SLACK_NODE_SET,
    )


async def remember_note(
    user_id: UUID,
    *,
    text: str,
    author_id: Optional[str] = None,
) -> None:
    """Store a free-text ``/cognee-remember`` note, attributed to ``user_id``.

    The shortcut above covers what Slack already has on screen; this covers
    what it never saw — a decision from a call, a conclusion someone records
    after the fact. Same dataset and node_set on purpose, so both arrive as
    one Slack-origin body of memory rather than two the graph can't relate.
    """
    owner = await get_user(user_id)
    await cognee_remember(
        _format_noted_text(text, author_id=author_id),
        dataset_name=SLACK_DATASET_NAME,
        user=owner,
        run_in_background=True,
        node_set=SLACK_NODE_SET,
    )
