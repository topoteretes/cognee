"""Unit tests for cognee.modules.integrations.slack.remember_message.

Invariants: the enriched text carries channel/author provenance (falling
back gracefully when either is missing), the Slack dataset name is used,
the source node_set is stamped so the origin survives into the graph,
and the background mode is always requested (no LLM work should block the
3-second interactive-payload window). Every item records its workspace, a
shortcut's item also its message: one message is one item per user, and what a
user remembered from a workspace can be found again to be forgotten.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.integrations.slack.remember_message import (
    CHANNEL_KEY,
    SLACK_DATASET_NAME,
    SLACK_NODE_SET,
    TEAM_KEY,
    TS_KEY,
    forget_remembered,
    forget_remembered_message,
    remember_message,
    remember_note,
    remembered_message_id,
)

MODULE = "cognee.modules.integrations.slack.remember_message"

MESSAGE = {"team_id": "T1", "channel_id": "C1", "message_ts": "1700000000.000100"}


@pytest.mark.asyncio
async def test_remembers_with_channel_and_author():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(
            owner.id, text="ship it", channel_name="general", author_id="U1", **MESSAGE
        )

    remember.assert_awaited_once()
    (item,), kwargs = remember.call_args
    assert item.data == "In #general, <@U1> said: ship it"
    assert kwargs["dataset_name"] == SLACK_DATASET_NAME
    assert kwargs["user"] is owner
    assert kwargs["run_in_background"] is True
    # Origin stamp: without this, a Slack-sourced document reaches the graph
    # with no structured Slack marker at all (only the dataset name and the
    # prose prefix), so it can't be grouped, colored, or filtered by source.
    assert kwargs["node_set"] == SLACK_NODE_SET == ["slack"]


@pytest.mark.asyncio
async def test_falls_back_when_channel_and_author_are_missing():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(owner.id, text="ship it", **MESSAGE)

    (item,), _ = remember.call_args
    assert item.data == "In Slack, someone said: ship it"


@pytest.mark.asyncio
async def test_the_item_records_its_workspace_channel_and_message():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_message(owner.id, text="ship it", **MESSAGE)

    (item,), _ = remember.call_args
    assert item.external_metadata == {
        TEAM_KEY: "T1",
        CHANNEL_KEY: "C1",
        TS_KEY: "1700000000.000100",
    }
    assert item.data_id == remembered_message_id(owner.id, **MESSAGE)
    # Slack text is never read as a URL or a file path.
    assert item.literal_text is True


def test_one_message_is_one_item_per_user():
    ana, ben = uuid4(), uuid4()
    other_message = {**MESSAGE, "message_ts": "1700000000.000200"}

    assert remembered_message_id(ana, **MESSAGE) == remembered_message_id(ana, **MESSAGE)
    assert remembered_message_id(ana, **MESSAGE) != remembered_message_id(ana, **other_message)
    # Data.id is global and a pinned id of another user's dataset is refused, so two
    # members remembering one message must not share an id.
    assert remembered_message_id(ana, **MESSAGE) != remembered_message_id(ben, **MESSAGE)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["team_id", "channel_id", "message_ts"])
async def test_a_message_without_its_team_channel_or_ts_is_refused(missing):
    with (
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
        pytest.raises(ValueError),
    ):
        await remember_message(uuid4(), text="ship it", **{**MESSAGE, missing: ""})

    remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_note_is_the_item_of_the_message_that_asked_for_it():
    """A note is found again by its Slack message, like a remembered message: `@cognee
    forget` in its thread, "Forget this" on it, and deleting the message all reach it."""
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_note(owner.id, text="we chose Neon", author_id="U1", **MESSAGE)

    (item,), kwargs = remember.call_args
    assert item.data == "In Slack, <@U1> noted: we chose Neon"
    assert item.external_metadata == {
        TEAM_KEY: "T1",
        CHANNEL_KEY: "C1",
        TS_KEY: MESSAGE["message_ts"],
    }
    assert item.data_id == remembered_message_id(owner.id, **MESSAGE)
    assert item.literal_text is True
    assert kwargs["dataset_name"] == SLACK_DATASET_NAME


@pytest.mark.asyncio
async def test_the_same_text_in_two_messages_is_two_notes():
    owner = SimpleNamespace(id=uuid4())
    with (
        patch(f"{MODULE}.get_user", new=AsyncMock(return_value=owner)),
        patch(f"{MODULE}.cognee_remember", new=AsyncMock()) as remember,
    ):
        await remember_note(owner.id, text="we chose Neon", **MESSAGE)
        await remember_note(owner.id, text="we chose Neon", **MESSAGE)
        await remember_note(
            owner.id, text="we chose Neon", **{**MESSAGE, "message_ts": "1700000000.000900"}
        )

    first, again, other = [call.args[0].data_id for call in remember.await_args_list]
    assert first == again != other


@pytest.mark.asyncio
async def test_forget_remembered_forgets_only_the_workspaces_items():
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4())
    here, elsewhere, unstamped, no_metadata = uuid4(), uuid4(), uuid4(), uuid4()
    rows = [
        SimpleNamespace(id=here, external_metadata={TEAM_KEY: "T1"}),
        SimpleNamespace(id=elsewhere, external_metadata={TEAM_KEY: "T2"}),
        # Stored before items carried the workspace: nothing ties it to T1.
        SimpleNamespace(id=unstamped, external_metadata={"node_set": ["slack"]}),
        SimpleNamespace(id=no_metadata, external_metadata=None),
    ]
    with (
        patch(f"{MODULE}.get_datasets_by_name", new=AsyncMock(return_value=[dataset])) as by_name,
        patch(f"{MODULE}.get_dataset_data", new=AsyncMock(return_value=rows)),
        patch(f"{MODULE}.forget", new=AsyncMock()) as forget,
    ):
        await forget_remembered(user, team_id="T1")

    # Only the user's own dataset is looked up: what other members remembered is theirs.
    by_name.assert_awaited_once_with(SLACK_DATASET_NAME, user.id)
    forget.assert_awaited_once_with(data_id=here, dataset_id=dataset.id, user=user)


@pytest.mark.asyncio
async def test_forget_remembered_tries_every_item_before_it_raises():
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4())
    first, second = uuid4(), uuid4()
    rows = [
        SimpleNamespace(id=first, external_metadata={TEAM_KEY: "T1"}),
        SimpleNamespace(id=second, external_metadata={TEAM_KEY: "T1"}),
    ]
    tried = []

    async def fake_forget(*, data_id, dataset_id, user):
        tried.append(data_id)
        if data_id == first:
            raise RuntimeError("store down")

    with (
        patch(f"{MODULE}.get_datasets_by_name", new=AsyncMock(return_value=[dataset])),
        patch(f"{MODULE}.get_dataset_data", new=AsyncMock(return_value=rows)),
        patch(f"{MODULE}.forget", new=fake_forget),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await forget_remembered(user, team_id="T1")

    assert tried == [first, second]


@pytest.mark.asyncio
async def test_forget_remembered_with_no_slack_dataset_does_nothing():
    with (
        patch(f"{MODULE}.get_datasets_by_name", new=AsyncMock(return_value=[])),
        patch(f"{MODULE}.forget", new=AsyncMock()) as forget,
    ):
        await forget_remembered(SimpleNamespace(id=uuid4()), team_id="T1")

    forget.assert_not_awaited()


def _engine(candidates):
    """A relational engine whose session answers the owners of ``slack`` datasets."""

    @asynccontextmanager
    async def session():
        yield SimpleNamespace(
            scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: list(candidates)))
        )

    return SimpleNamespace(get_async_session=session)


@pytest.mark.asyncio
async def test_forget_remembered_message_forgets_every_users_copy():
    ana, ben, cy = uuid4(), uuid4(), uuid4()
    message = {"channel_id": "C1", "message_ts": "1700000000.000100"}
    ids = {user: remembered_message_id(user, team_id="T1", **message) for user in (ana, ben, cy)}
    users = {user: SimpleNamespace(id=user) for user in ids}
    with (
        patch(f"{MODULE}.get_relational_engine", return_value=_engine(ids)),
        patch(
            f"{MODULE}.get_existing_data_ids", new=AsyncMock(return_value={ids[ana], ids[ben]})
        ) as existing,
        patch(f"{MODULE}.get_user", new=AsyncMock(side_effect=lambda user_id: users[user_id])),
        patch(f"{MODULE}.forget", new=AsyncMock()) as forget,
    ):
        await forget_remembered_message("T1", **message)

    # Every slack dataset owner's derived id is a candidate; only rows are forgotten.
    assert set(existing.await_args.args[1]) == set(ids.values())
    forgotten = {call.kwargs["data_id"]: call.kwargs for call in forget.await_args_list}
    assert set(forgotten) == {ids[ana], ids[ben]}
    assert forgotten[ids[ana]]["user"] is users[ana]
    assert forgotten[ids[ana]]["dataset"] == SLACK_DATASET_NAME


@pytest.mark.asyncio
async def test_forget_remembered_message_tries_every_copy_before_it_raises():
    ana, ben = uuid4(), uuid4()
    message = {"channel_id": "C1", "message_ts": "1700000000.000100"}
    ids = {user: remembered_message_id(user, team_id="T1", **message) for user in (ana, ben)}
    tried = []

    async def fake_forget(*, data_id, dataset, user):
        tried.append(data_id)
        if data_id == ids[ana]:
            raise RuntimeError("store down")

    with (
        patch(f"{MODULE}.get_relational_engine", return_value=_engine(ids)),
        patch(f"{MODULE}.get_existing_data_ids", new=AsyncMock(return_value=set(ids.values()))),
        patch(f"{MODULE}.get_user", new=AsyncMock(side_effect=lambda u: SimpleNamespace(id=u))),
        patch(f"{MODULE}.forget", new=fake_forget),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await forget_remembered_message("T1", **message)

    assert set(tried) == set(ids.values())


@pytest.mark.asyncio
async def test_forget_remembered_message_with_no_copy_forgets_nothing():
    with (
        patch(f"{MODULE}.get_relational_engine", return_value=_engine([uuid4()])),
        patch(f"{MODULE}.get_existing_data_ids", new=AsyncMock(return_value=set())),
        patch(f"{MODULE}.forget", new=AsyncMock()) as forget,
    ):
        await forget_remembered_message("T1", channel_id="C1", message_ts="1.1")

    forget.assert_not_awaited()
