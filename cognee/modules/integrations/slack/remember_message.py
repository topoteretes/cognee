"""Remember a Slack message shortcut's content in Cognee.

Wraps :func:`cognee.remember` with a Slack-specific dataset and
provenance-carrying text — bare message text alone loses who said it and
where, both useful context for the resulting knowledge graph.

Every remembered item also records which workspace it came from, and the
shortcut's items which message: the workspace so a disconnect can delete what the
disconnecting user remembered from that workspace, the message so the same message
is one item and can be found again by ``(team, channel, ts)``, which is how a message
edited or deleted in Slack is followed (:func:`forget_remembered_message`).
"""

import logging
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select

from cognee.api.v1.forget.forget import forget
from cognee.api.v1.remember.remember import remember as cognee_remember
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.data.methods.get_existing_data_ids import get_existing_data_ids
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_user
from cognee.modules.users.models import User
from cognee.tasks.ingestion.data_item import DataItem

logger = logging.getLogger(__name__)

SLACK_DATASET_NAME = "slack"

# Structured origin stamp: every integration that ingests data should tag it
# with a node_set naming the source system, so the origin survives into the
# graph (NodeSet node + belongs_to_set edges + source_node_set property)
# instead of living only in the dataset name and the prose of the text.
SLACK_NODE_SET = ["slack"]

# Keys in ``Data.external_metadata`` of every remembered item.
TEAM_KEY = "slack_team_id"
CHANNEL_KEY = "slack_channel_id"
TS_KEY = "slack_message_ts"


def _format_remembered_text(text: str, *, channel_name: str | None, author_id: str | None) -> str:
    channel = f"#{channel_name}" if channel_name else "Slack"
    author = f"<@{author_id}>" if author_id else "someone"
    return f"In {channel}, {author} said: {text}"


def _format_noted_text(text: str, *, author_id: str | None) -> str:
    """Provenance for a note nobody said in Slack — see :func:`remember_note`.

    Deliberately "noted", not "said": the shortcut path quotes a real
    message, this one records something that happened elsewhere, and
    flattening the two would put words in someone's mouth in the graph.
    """
    author = f"<@{author_id}>" if author_id else "someone"
    return f"In Slack, {author} noted: {text}"


def remembered_message_id(user_id: UUID, *, team_id: str, channel_id: str, message_ts: str) -> UUID:
    """The data id of one remembered message: one message, one item, per user.

    The user is part of the id because ``Data.id`` is global and a pinned id that
    already belongs to another user's dataset is refused: two members remembering
    the same message must each get their own item.
    """
    return uuid5(NAMESPACE_URL, f"slack-remembered:{user_id}:{team_id}:{channel_id}:{message_ts}")


async def remember_message(
    user_id: UUID,
    *,
    text: str,
    team_id: str,
    channel_id: str,
    message_ts: str,
    channel_name: str | None = None,
    author_id: str | None = None,
) -> None:
    """Store one Slack message as a Cognee memory, attributed to ``user_id``.

    The item is forgotten when Slack reports the message edited or deleted
    (:func:`forget_remembered_message`), and by its owner at any time.

    ``run_in_background=True`` returns once the item is stored, so the caller (the
    shortcut's detached save, which answers through ``response_url``) does not wait for
    the cognify that follows.
    """
    if not (team_id and channel_id and message_ts):
        # Without all three every message would share one id and overwrite the last.
        raise ValueError("A remembered Slack message needs its team, channel and ts.")
    owner = await get_user(user_id)
    enriched_text = _format_remembered_text(text, channel_name=channel_name, author_id=author_id)
    # literal_text: the text is whatever a Slack user typed, and the provenance prefix
    # does not stop ingestion from reading it as a path (``..`` collapses through it).
    item = DataItem(
        data=enriched_text,
        literal_text=True,
        external_metadata={TEAM_KEY: team_id, CHANNEL_KEY: channel_id, TS_KEY: message_ts},
        data_id=remembered_message_id(
            user_id, team_id=team_id, channel_id=channel_id, message_ts=message_ts
        ),
    )
    await cognee_remember(
        item,
        dataset_name=SLACK_DATASET_NAME,
        user=owner,
        run_in_background=True,
        node_set=SLACK_NODE_SET,
        # A host stores what it was given, as the sync does: improve() would run conflict
        # review and the global context index over the member's whole dataset per click.
        self_improvement=False,
    )


async def remember_note(
    user_id: UUID,
    *,
    text: str,
    team_id: str,
    channel_id: str,
    message_ts: str,
    author_id: str | None = None,
) -> None:
    """Store a free-text ``@cognee remember`` note, attributed to ``user_id``.

    The shortcut above covers what Slack already has on screen; this covers
    what it never saw — a decision from a call, a conclusion someone records
    after the fact. Same dataset and node_set on purpose, so both arrive as
    one Slack-origin body of memory rather than two the graph can't relate.
    A note is the item of the message that asked for it (Goran, 2026-11-10): it is found
    again by that message, so ``@cognee forget`` in its thread, "Forget this" on it and
    deleting the message in Slack all forget it, as for a remembered message. The same
    text noted twice is two notes.
    """
    if not (team_id and channel_id and message_ts):
        raise ValueError("A Slack note needs the team, channel and ts of its message.")
    owner = await get_user(user_id)
    noted = _format_noted_text(text, author_id=author_id)
    item = DataItem(
        data=noted,
        literal_text=True,
        external_metadata={TEAM_KEY: team_id, CHANNEL_KEY: channel_id, TS_KEY: message_ts},
        data_id=remembered_message_id(
            user_id, team_id=team_id, channel_id=channel_id, message_ts=message_ts
        ),
    )
    await cognee_remember(
        item,
        dataset_name=SLACK_DATASET_NAME,
        user=owner,
        run_in_background=True,
        node_set=SLACK_NODE_SET,
        # A host stores what it was given, as the sync does: improve() would run conflict
        # review and the global context index over the member's whole dataset per click.
        self_improvement=False,
    )


async def forget_remembered(user: User, *, team_id: str) -> None:
    """Forget what ``user`` remembered from the workspace ``team_id``.

    Only the user's own ``slack`` dataset is read: what other members remembered
    belongs to them. Items stored before they carried the workspace have no
    ``slack_team_id`` and are left. Every item is tried before the first failure is
    raised, so one item that cannot be forgotten does not keep the others.
    """
    failures: list[Exception] = []
    for dataset in await get_datasets_by_name(SLACK_DATASET_NAME, user.id):
        for data in await get_dataset_data(dataset.id):
            if (data.external_metadata or {}).get(TEAM_KEY) != team_id:
                continue
            try:
                await forget(data_id=data.id, dataset_id=dataset.id, user=user)
            except Exception as error:  # keep forgetting the other items, then report
                logger.exception("Could not forget a remembered Slack item")
                failures.append(error)
    if failures:
        raise failures[0]


async def forget_remembered_message(team_id: str, *, channel_id: str, message_ts: str) -> None:
    """Forget every copy of one Slack message, for each user who remembered it.

    Slack reported the message deleted, or edited: what was remembered may be what its
    author took back. Whoever remembered a message owns a ``slack`` dataset, and a copy's
    id derives from ``(user, team, channel, ts)``, so the owners of those datasets are the
    candidates and one primary-key lookup finds the copies: no index to keep in step, and
    no dependence on who holds the workspace's connection now (a previous owner's copies
    are followed too). Every copy is tried before the first failure is raised, so Slack's
    retry of the event finds only what is left; forgetting twice is harmless.
    """
    async with get_relational_engine().get_async_session() as session:
        candidates = await _remembering_user_ids(session)
        owners = {
            remembered_message_id(
                user_id, team_id=team_id, channel_id=channel_id, message_ts=message_ts
            ): user_id
            for user_id in candidates
        }
        remembered = await get_existing_data_ids(session, list(owners))

    failures: list[Exception] = []
    for data_id in remembered:
        try:
            user = await get_user(owners[data_id])
            await forget(data_id=data_id, dataset=SLACK_DATASET_NAME, user=user)
        except Exception as error:  # keep forgetting the other copies, then report
            logger.exception("Could not forget a remembered Slack message")
            failures.append(error)
    if failures:
        raise failures[0]


async def forget_my_remembered_message(
    user_id: UUID, *, team_id: str, channel_id: str, message_ts: str
) -> bool:
    """Forget ``user_id``'s own copy of one Slack message; ``False`` when there was none.

    ``@cognee forget`` and "Forget this". The copy's id derives from the user and the
    message, so this reaches only the asking member's copy; others who remembered the
    same message keep theirs.
    """
    data_id = remembered_message_id(
        user_id, team_id=team_id, channel_id=channel_id, message_ts=message_ts
    )
    async with get_relational_engine().get_async_session() as session:
        existing = await get_existing_data_ids(session, [data_id])
    if data_id not in existing:
        return False
    await forget(data_id=data_id, dataset=SLACK_DATASET_NAME, user=await get_user(user_id))
    return True


async def forget_workspace_remembered(team_id: str) -> None:
    """Forget what every user remembered from the workspace ``team_id``.

    For a workspace whose data is deleted for everyone (an uninstall, G28). The users who
    remembered anything are the owners of ``slack`` datasets; each one's items from the
    workspace go through :func:`forget_remembered`. Every user is tried before the first
    failure is raised.
    """
    async with get_relational_engine().get_async_session() as session:
        user_ids = await _remembering_user_ids(session)

    failures: list[Exception] = []
    for user_id in user_ids:
        try:
            await forget_remembered(await get_user(user_id), team_id=team_id)
        except Exception as error:  # keep forgetting the other users' copies, then report
            logger.exception("Could not forget a user's remembered Slack items")
            failures.append(error)
    if failures:
        raise failures[0]


async def _remembering_user_ids(session) -> list[UUID]:
    """Everyone who remembered anything from Slack: the owners of ``slack`` datasets."""
    return list(
        (
            await session.scalars(
                select(Dataset.owner_id).where(Dataset.name == SLACK_DATASET_NAME).distinct()
            )
        ).all()
    )
