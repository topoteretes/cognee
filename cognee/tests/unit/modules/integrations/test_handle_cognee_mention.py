"""``@cognee``: a mention in a channel, or a direct message to the app, is the one way to
talk to Cognee from Slack (``@cognee <question>``, ``@cognee remember …``,
``@cognee link``, ``@cognee help``). An answer is a message (``post_message``): in a
channel's thread for everyone there, or in the direct message; every other reply is
ephemeral (``post_ephemeral``). Slack is otherwise stood in for by ``FakeSlack``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.integrations.slack import channel_membership as membership
from cognee.modules.integrations.slack import handle_cognee_mention as mention
from cognee.modules.integrations.slack import handle_slack_event as events
from cognee.modules.integrations.slack.handle_slack_link import NOT_LINKED_MESSAGE
from cognee.modules.integrations.slack.persistence import CHANNEL_NOT_ENABLED_MESSAGE
from cognee.tests.unit.tasks.connectors.fake_slack import FakeSlack

TEAM = "T0TEAM"
BOT = "U0BOT"
MEMBER = "U0MEMBER"
OWNER_ID = "11111111-1111-1111-1111-111111111111"
WORKSPACE_DATASET = mention.dataset_name_for_credential(SimpleNamespace(provider_account_id=TEAM))


def _credential(allowed=None, synced=()):
    metadata = {
        "installed_by_slack_user_id": "U0INSTALLER",
        "bot_user_id": BOT,
        "selected_channels": list(synced),
    }
    if allowed is not None:
        metadata["allowed_channel_ids"] = allowed
    return SimpleNamespace(
        status="active",
        provider_account_id=TEAM,
        user_id=OWNER_ID,
        provider_metadata=metadata,
        last_synced_at=None,
        sync_status=None,
    )


def _mention(text: str, channel: str = "C1", **extra) -> dict:
    return {"type": "app_mention", "user": MEMBER, "channel": channel, "ts": "1.1", **extra} | {
        "text": f"<@{BOT}> {text}".strip()
    }


@pytest.fixture
def slack(monkeypatch):
    """The connected workspace, the member's link, the channels Slack says the member is in,
    the channels Slack says are public, whether the channel is shared with another
    organization, the datasets the member can read (the workspace's and one of their own),
    every ephemeral reply and every answer posted."""
    state = SimpleNamespace(
        credential=_credential(),
        owner=OWNER_ID,
        replies=[],
        threads=[],
        channels={"C1", "C7"},
        public={"C1", "C3"},
        external=False,
        posted=[],
        workspace=SimpleNamespace(id=uuid4(), name=WORKSPACE_DATASET),
        own=SimpleNamespace(id=uuid4(), name="notes"),
    )

    async def post_ephemeral(credential, channel, user, payload, thread_ts=None):
        state.replies.append((channel, user, payload))
        state.threads.append(thread_ts)

    monkeypatch.setattr(
        mention, "get_by_team", AsyncMock(side_effect=lambda team: state.credential)
    )
    monkeypatch.setattr(
        mention, "resolve_owner_user_id", AsyncMock(side_effect=lambda *args: state.owner)
    )

    async def post_message(credential, channel, payload, thread_ts=None):
        state.posted.append((channel, thread_ts, payload))

    monkeypatch.setattr(mention, "post_ephemeral", post_ephemeral)
    monkeypatch.setattr(mention, "post_message", post_message)
    monkeypatch.setattr(
        membership, "public_channels", AsyncMock(side_effect=lambda *a: state.public)
    )
    monkeypatch.setattr(
        membership,
        "shared_with_other_organizations",
        AsyncMock(side_effect=lambda *args: state.external),
    )
    monkeypatch.setattr(
        mention, "member_channels", AsyncMock(side_effect=lambda *args: state.channels)
    )
    monkeypatch.setattr(
        mention,
        "get_authorized_existing_datasets",
        AsyncMock(side_effect=lambda *args: [state.workspace, state.own]),
    )
    return state


def _texts(state) -> list[str]:
    return [payload["text"] for _, _, payload in state.replies]


def _direct(text: str) -> dict:
    """A direct message to the app."""
    return {"type": "message", "channel_type": "im", "user": MEMBER, "channel": "D1"} | {
        "text": text,
        "ts": "2.2",
    }


@pytest.mark.asyncio
async def test_link_answers_the_sign_in_page_privately(slack):
    callback = "https://app/api/v1/integrations/slack_member/callback"
    with patch(
        "cognee.modules.integrations.slack.handle_slack_link.require", return_value=callback
    ):
        await mention.handle_cognee_mention(TEAM, _mention("link"))

    ((channel, user, payload),) = slack.replies
    assert (channel, user) == ("C1", MEMBER)
    text = payload["text"]
    assert "https://app/api/v1/integrations/slack_member/start" in text and "code=" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "help", "HELP"])
async def test_a_bare_mention_or_help_explains_the_commands(slack, text):
    await mention.handle_cognee_mention(TEAM, _mention(text))

    (reply,) = _texts(slack)
    assert "@cognee remember" in reply and "@cognee link" in reply


@pytest.mark.asyncio
async def test_a_member_who_never_linked_is_told_how(slack):
    slack.owner = None
    with patch.object(mention, "ask", AsyncMock()) as ask:
        await mention.handle_cognee_mention(TEAM, _mention("why Neon?"))

    assert _texts(slack) == [NOT_LINKED_MESSAGE]
    ask.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_channel_outside_the_allowlist_is_refused_but_a_direct_message_is_not(slack):
    slack.credential = _credential(allowed=["C9"])
    with patch.object(mention, "ask", AsyncMock()) as ask:
        await mention.handle_cognee_mention(TEAM, _mention("why Neon?"))
        assert _texts(slack) == [CHANNEL_NOT_ENABLED_MESSAGE]

        direct = {"type": "message", "channel_type": "im", "user": MEMBER, "channel": "D1"}
        await mention.handle_cognee_mention(TEAM, direct | {"text": "why Neon?", "ts": "2.2"})

    ask.assert_awaited_once()
    assert ask.await_args.kwargs["question"] == "why Neon?"


@pytest.mark.asyncio
async def test_remember_with_text_saves_a_note_for_the_member(slack):
    with patch.object(mention, "remember_note", AsyncMock()) as save:
        await mention.handle_cognee_mention(TEAM, _mention("remember we chose Neon for v2"))

    save.assert_awaited_once()
    assert save.await_args.kwargs == {
        "text": "we chose Neon for v2",
        "team_id": TEAM,
        "channel_id": "C1",
        "message_ts": "1.1",  # the note is the item of this message
        "author_id": MEMBER,
    }
    assert _texts(slack)[-1] == mention.REMEMBERED_MESSAGE


@pytest.mark.asyncio
async def test_remember_in_a_thread_saves_the_threads_first_message(slack):
    fake = FakeSlack()
    fake.add_channel("C1", "general")
    fake.post("1700000000.000100", "U0AUTHOR", "The launch moved to Friday.", channel="C1")
    with (
        patch.object(mention.ingestion, "build_service", lambda provider, token: fake),
        patch.object(mention, "access_token_for", AsyncMock(return_value="xoxb")),
        patch.object(mention, "app_follows_channel", AsyncMock(return_value=True)),
        patch.object(mention, "remember_message", AsyncMock()) as save,
    ):
        await mention.handle_cognee_mention(
            TEAM, _mention("remember", thread_ts="1700000000.000100", ts="1700000099.000100")
        )

    save.assert_awaited_once()
    assert save.await_args.kwargs | {"channel_name": None} == {
        "text": "The launch moved to Friday.",
        "team_id": TEAM,
        "channel_id": "C1",
        "message_ts": "1700000000.000100",
        "channel_name": None,
        "author_id": "U0AUTHOR",
    }


@pytest.mark.asyncio
async def test_a_workspace_that_is_not_connected_gets_no_answer(slack):
    slack.credential = None
    await mention.handle_cognee_mention(TEAM, _mention("why Neon?"))
    assert slack.replies == []


def _envelope(event: dict) -> bytes:
    return json.dumps({"type": "event_callback", "team_id": TEAM, "event": event}).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "handled"),
    [
        (_mention("why Neon?"), True),
        (
            {
                "type": "message",
                "channel_type": "im",
                "user": MEMBER,
                "channel": "D1",
                "text": "hi",
            },
            True,
        ),
        # The app's own messages in the DM come back as events too: never answered.
        (
            {
                "type": "message",
                "channel_type": "im",
                "bot_id": "B1",
                "channel": "D1",
                "text": "hi",
            },
            False,
        ),
        # An ordinary channel message is only followed for remembered copies.
        ({"type": "message", "channel_type": "channel", "user": MEMBER, "channel": "C1"}, False),
    ],
    ids=["mention", "direct_message", "the_apps_own_message", "channel_message"],
)
async def test_the_event_handler_answers_at_once_and_handles_mentions_and_direct_messages(
    event, handled
):
    started = []
    with (
        patch.object(events, "handle_cognee_mention", AsyncMock()) as handle,
        patch.object(events, "register_background_task", side_effect=started.append),
        patch.object(events, "forget_remembered_message", AsyncMock()),
    ):
        assert await events.handle_slack_event(_envelope(event)) == {"ok": True}
        for task in started:
            await task

    assert handle.await_count == (1 if handled else 0)


@pytest.mark.asyncio
async def test_a_failed_search_is_reported_instead_of_raised(slack):
    slack.credential = _credential(synced=["C1"])
    with (
        patch.object(mention, "cognee_search", AsyncMock(side_effect=RuntimeError("boom"))),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("why Neon?"))

    assert _texts(slack)[-1] == mention.SEARCH_FAILED_MESSAGE


@pytest.mark.asyncio
async def test_a_connection_whose_owner_was_deleted_says_so(slack):
    from cognee.infrastructure.databases.exceptions import EntityNotFoundError

    with patch.object(mention, "get_user", AsyncMock(side_effect=EntityNotFoundError("gone"))):
        await mention.handle_cognee_mention(TEAM, _mention("why Neon?"))

    assert _texts(slack)[-1] == mention.NOT_CONFIGURED_MESSAGE


@pytest.mark.asyncio
async def test_a_failed_save_is_reported_instead_of_raised(slack):
    with patch.object(mention, "remember_note", AsyncMock(side_effect=RuntimeError("boom"))):
        await mention.handle_cognee_mention(TEAM, _mention("remember we chose Neon"))

    assert _texts(slack)[-1] == mention.SAVE_FAILED_MESSAGE


@pytest.mark.asyncio
async def test_forget_in_a_thread_forgets_the_members_copy_of_its_first_message(slack):
    with patch.object(
        mention, "forget_my_remembered_message", AsyncMock(return_value=True)
    ) as forget:
        await mention.handle_cognee_mention(
            TEAM, _mention("forget", thread_ts="1700000000.000100", ts="1700000099.000100")
        )

    forget.assert_awaited_once_with(
        OWNER_ID, team_id=TEAM, channel_id="C1", message_ts="1700000000.000100"
    )
    assert _texts(slack)[-1] == mention.FORGOTTEN_MESSAGE


@pytest.mark.asyncio
async def test_forget_says_so_when_there_was_no_copy(slack):
    with patch.object(mention, "forget_my_remembered_message", AsyncMock(return_value=False)):
        await mention.handle_cognee_mention(TEAM, _mention("forget", thread_ts="1.1"))

    assert _texts(slack)[-1] == mention.NOTHING_TO_FORGET_MESSAGE


@pytest.mark.asyncio
async def test_forget_outside_a_thread_explains_where_to_use_it(slack):
    with patch.object(mention, "forget_my_remembered_message", AsyncMock()) as forget:
        await mention.handle_cognee_mention(TEAM, _mention("forget"))

    forget.assert_not_awaited()
    assert "thread" in _texts(slack)[-1] and "Forget this" in _texts(slack)[-1]


@pytest.mark.asyncio
async def test_forgetting_works_in_a_channel_outside_the_allowlist(slack):
    """The allowlist limits where Cognee is used; taking one's own copy back is never limited."""
    slack.credential = _credential(allowed=["C9"])
    with patch.object(
        mention, "forget_my_remembered_message", AsyncMock(return_value=True)
    ) as forget:
        await mention.handle_cognee_mention(TEAM, _mention("forget", thread_ts="1.1"))

    forget.assert_awaited_once()


@pytest.mark.asyncio
async def test_forget_in_a_direct_message_thread_forgets_a_note_made_there(slack):
    direct = {"type": "message", "channel_type": "im", "user": MEMBER, "channel": "D1"}
    with patch.object(
        mention, "forget_my_remembered_message", AsyncMock(return_value=True)
    ) as forget:
        await mention.handle_cognee_mention(
            TEAM,
            direct | {"text": "forget", "thread_ts": "1700000000.000100", "ts": "1700000099.1"},
        )

    forget.assert_awaited_once_with(
        OWNER_ID, team_id=TEAM, channel_id="D1", message_ts="1700000000.000100"
    )
    assert _texts(slack)[-1] == mention.FORGOTTEN_MESSAGE


def _synced_thread(author: str) -> FakeSlack:
    fake = FakeSlack()
    fake.add_channel("C1", "general")
    fake.post("1700000000.000100", author, "Lunch is at noon, ignore.", channel="C1")
    return fake


async def _forget_synced(slack, fake):
    with (
        patch.object(mention.ingestion, "build_service", lambda provider, token: fake),
        patch.object(mention, "access_token_for", AsyncMock(return_value="xoxb")),
        patch.object(mention, "forget_my_remembered_message", AsyncMock(return_value=False)),
        patch.object(mention, "exclude_message", AsyncMock(return_value=True)) as exclude,
    ):
        await mention.handle_cognee_mention(
            TEAM, _mention("forget", thread_ts="1700000000.000100", ts="1700000099.000100")
        )
    return exclude


@pytest.mark.asyncio
async def test_the_author_takes_a_synced_message_out_of_the_workspace_memory(slack):
    """Goran, 2026-11-10: the author or the connection's owner may take a message out of
    the shared memory; the sync then leaves it out for good."""
    slack.credential = _credential(synced=["C1"])
    slack.owner = "22222222-2222-2222-2222-222222222222"  # a linked member, not the owner
    exclude = await _forget_synced(slack, _synced_thread(MEMBER))

    exclude.assert_awaited_once_with(
        slack.credential,
        channel_id="C1",
        message_ts="1700000000.000100",
        root_ts="1700000000.000100",
    )
    assert _texts(slack)[-1] == mention.EXCLUDED_MESSAGE


@pytest.mark.asyncio
async def test_the_owner_takes_anyones_synced_message_out(slack):
    slack.credential = _credential(synced=["C1"])  # the member resolves to the owner
    exclude = await _forget_synced(slack, _synced_thread("U0SOMEONE"))

    exclude.assert_awaited_once()
    assert _texts(slack)[-1] == mention.EXCLUDED_MESSAGE


@pytest.mark.asyncio
async def test_another_member_cannot_take_someone_elses_message_out(slack):
    slack.credential = _credential(synced=["C1"])
    slack.owner = "22222222-2222-2222-2222-222222222222"
    exclude = await _forget_synced(slack, _synced_thread("U0SOMEONE"))

    exclude.assert_not_awaited()
    assert _texts(slack)[-1] == mention.NOT_YOURS_TO_EXCLUDE_MESSAGE


@pytest.mark.asyncio
async def test_a_mention_in_a_thread_is_answered_in_that_thread(slack):
    """Reviewer C's F14: answered without ``thread_ts``, the reply lands in the channel
    view, so a member who wrote `@cognee forget` in a thread never sees "Forgotten"."""
    with patch.object(mention, "forget_my_remembered_message", AsyncMock(return_value=True)):
        await mention.handle_cognee_mention(TEAM, _mention("forget", thread_ts="1.1", ts="1.2"))
        await mention.handle_cognee_mention(TEAM, _mention("help"))

    assert slack.threads == ["1.1", None], "in the thread, and in the channel outside one"


@pytest.mark.asyncio
async def test_post_ephemeral_sends_the_thread_to_slack(monkeypatch):
    sent = {}

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {"ok": True}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None, allow_redirects=True):
            sent.update(json)
            return _Response()

    monkeypatch.setattr(mention, "client_session", lambda provider, **kwargs: _Session())
    monkeypatch.setattr(mention, "access_token_for", AsyncMock(return_value="xoxb"))

    await mention.post_ephemeral(_credential(), "C1", MEMBER, {"text": "hi"}, thread_ts="1.1")

    assert sent == {
        "channel": "C1",
        "user": MEMBER,
        "text": "hi",
        "unfurl_links": False,
        "unfurl_media": False,
        "thread_ts": "1.1",
    }


def test_every_command_in_the_help_is_one_the_sync_leaves_out_of_memory():
    """One list of command words: what `@cognee` answers is what the channel sync skips."""
    import re

    from cognee.tasks.ingestion.connectors.slack import APP_COMMANDS

    words = set(re.findall(r"`@cognee (\w+)", mention.HELP_MESSAGE)) - {"what"}
    assert words and words <= APP_COMMANDS, words - APP_COMMANDS


def test_a_reply_is_a_bold_title_over_one_line_and_progress_is_a_grey_line():
    """Goran, 2026-11-10: replies follow cognee's design, a short title and a plain line,
    no emoji."""
    import re

    reply = mention.notice(mention.FORGOTTEN_MESSAGE)
    assert reply["blocks"] == [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Forgotten*\nCognee no longer has your copy of this message.",
            },
        }
    ]
    assert mention.notice("Searching", muted=True)["blocks"][0]["type"] == "context"
    emoji = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")
    texts = [value for name, value in vars(mention).items() if name.endswith("_MESSAGE")]
    assert texts and not [text for text in texts if emoji.search(text)]


# --- unlink ---


@pytest.mark.asyncio
async def test_unlink_ends_the_members_own_link(slack):
    with (
        patch.object(
            mention,
            "get_credential_by_account",
            AsyncMock(return_value=SimpleNamespace(status="active")),
        ),
        patch.object(
            mention, "revoke_credential_by_account", AsyncMock(return_value=True)
        ) as revoke,
    ):
        await mention.handle_cognee_mention(TEAM, _mention("unlink"))

    revoke.assert_awaited_once_with(
        "slack_member", f"{TEAM}:{MEMBER}", revoked_by=mention.REVOKED_BY_OWNER
    )
    assert _texts(slack)[-1] == mention.UNLINKED_MESSAGE


@pytest.mark.asyncio
async def test_unlink_without_a_link_changes_nothing(slack):
    with (
        patch.object(mention, "get_credential_by_account", AsyncMock(return_value=None)),
        patch.object(mention, "revoke_credential_by_account", AsyncMock()) as revoke,
    ):
        await mention.handle_cognee_mention(TEAM, _mention("unlink"))

    revoke.assert_not_awaited()
    assert _texts(slack)[-1] == mention.NOT_LINKED_TO_UNLINK_MESSAGE


@pytest.mark.asyncio
async def test_the_workspace_dataset_is_searched_only_in_the_selected_channels_the_member_is_in(
    slack,
):
    """F4: the workspace dataset holds every selected channel, private ones included; a
    direct message is answered from it only within the channels Slack says the member
    belongs to, and from their own datasets whole."""
    slack.credential = _credential(synced=["C1", "G2", "C3"])
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _direct("when do we deploy?"))

    calls = [(c.kwargs["dataset_ids"], c.kwargs["node_name"]) for c in search.await_args_list]
    assert calls == [
        ([slack.workspace.id], ["slack:channel:C1"]),
        ([slack.own.id], None),
    ]


@pytest.mark.asyncio
async def test_when_slack_cannot_say_where_the_member_is_the_workspace_is_left_out(slack):
    """F4: an unknown membership fails closed: the workspace dataset is not searched, the
    member is told, and their own datasets still answer."""
    from cognee.tasks.ingestion.connectors.slack import SlackUnavailableError

    slack.credential = _credential(synced=["C1"])
    with (
        patch.object(
            mention,
            "member_channels",
            AsyncMock(side_effect=SlackUnavailableError("users.conversations", "http_503")),
        ),
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _direct("when do we deploy?"))

    assert [c.kwargs["dataset_ids"] for c in search.await_args_list] == [[slack.own.id]]
    assert mention.MEMBERSHIP_UNKNOWN_MESSAGE in _texts(slack)


# -- Reprobe 2026-11-12 (N3) --------------------------------------------------------------


@pytest.mark.asyncio
async def test_any_failure_to_learn_the_members_channels_leaves_the_workspace_out(slack):
    """N3: only ``SlackSourceError`` was caught around the membership lookup, so any other
    failure on the way (outbound HTTP switched off, a token that cannot be read) ended the
    question silently. It fails closed the same way: the member is told and their own
    datasets still answer."""
    slack.credential = _credential(synced=["C1"])
    with (
        patch.object(mention, "member_channels", AsyncMock(side_effect=RuntimeError("no token"))),
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _direct("when do we deploy?"))

    assert [c.kwargs["dataset_ids"] for c in search.await_args_list] == [[slack.own.id]]
    assert mention.MEMBERSHIP_UNKNOWN_MESSAGE in _texts(slack)


# -- Public answers in a channel (SDK-1502) ----------------------------------------------
#
# Goran, 2026-11-12: a question asked in a channel is answered in its thread, for everyone
# in the channel, so the answer may use only what every member there can already see: that
# channel and the selected public channels, or that channel alone when it is shared with
# another organization. A direct message is answered with a message in it, from today's
# scope.

_FOUND = [{"text_result": ["We chose Neon for branching."], "context_result": "Neon chunk"}]


def _scopes(search) -> list[tuple]:
    """Each search as (datasets, sorted node names): order inside a scope means nothing."""
    return [
        (call.kwargs["dataset_ids"], sorted(call.kwargs["node_name"] or []) or None)
        for call in search.await_args_list
    ]


def _actions(payload: dict) -> list[dict]:
    return [block for block in payload.get("blocks", []) if block["type"] == "actions"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "where, thread",
    [({}, "1.1"), ({"thread_ts": "0.9"}, "0.9")],
    ids=["under the question", "in the thread it was asked in"],
)
async def test_a_channel_question_is_answered_publicly_in_its_thread(slack, where, thread):
    """Everyone in the channel sees the answer, next to the question: in a new thread under
    it, or in the thread it was asked in."""
    slack.credential = _credential(synced=["C1"])
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=_FOUND)),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("why did we choose Neon?", **where))

    assert len(slack.posted) == 1, "one answer, posted to the channel"
    ((channel, thread_ts, answer),) = slack.posted
    assert (channel, thread_ts) == ("C1", thread)
    assert "branching" in answer["text"]
    assert not [p for _, _, p in slack.replies if "branching" in p["text"]], "not twice"


@pytest.mark.asyncio
async def test_a_channel_answer_uses_this_channel_and_the_selected_public_channels_only(slack):
    """Asked in the private channel G4: G4 itself (everyone there reads it) and the selected
    public channels C1 and C3. G2 is private and the asker is in it, but others in G4 may
    not be; the asker's own datasets are theirs alone. Who the asker is does not change a
    public answer, so their channel membership is not asked for."""
    slack.credential = _credential(synced=["C1", "G2", "C3", "G4"])
    slack.channels = {"C1", "G2", "G4"}
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
        patch.object(mention, "member_channels", AsyncMock(return_value={"G2"})) as membership,
    ):
        await mention.handle_cognee_mention(TEAM, _mention("when do we deploy?", channel="G4"))

    assert _scopes(search) == [
        ([slack.workspace.id], ["slack:channel:C1", "slack:channel:C3", "slack:channel:G4"])
    ]
    membership.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_channel_answer_leaves_out_a_public_channel_that_is_not_selected(slack):
    """A public channel that is not synced has nothing in the workspace dataset; asking in
    one still answers from the selected public channels."""
    slack.credential = _credential(synced=["C3", "G2"])
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("when do we deploy?", channel="C1"))

    assert _scopes(search) == [([slack.workspace.id], ["slack:channel:C3"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("asking", ["public_channels", "shared_with_other_organizations"])
async def test_when_slack_cannot_describe_the_channels_only_this_channel_answers(slack, asking):
    """An unknown answer fails closed: a channel Slack did not call public, or a channel it
    did not say is unshared, gets no other channel. The member is told."""
    from cognee.tasks.ingestion.connectors.slack import SlackUnavailableError

    slack.credential = _credential(synced=["C1", "G2", "C3"])
    unknown = AsyncMock(side_effect=SlackUnavailableError("conversations.info", "http_503"))
    with (
        patch.object(membership, asking, unknown),
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("when do we deploy?", channel="C1"))

    assert _scopes(search) == [([slack.workspace.id], ["slack:channel:C1"])]
    assert mention.PUBLIC_CHANNELS_UNKNOWN_MESSAGE in _texts(slack)


@pytest.mark.asyncio
async def test_a_slack_connect_channel_is_answered_from_itself_alone(slack):
    """Goran, 2026-11-12: the other organization's members of a shared channel cannot read
    this workspace's public channels, so the answer there uses that channel only."""
    slack.credential = _credential(synced=["C1", "C3"])
    slack.external = True
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("when do we deploy?", channel="C1"))

    assert _scopes(search) == [([slack.workspace.id], ["slack:channel:C1"])]
    assert mention.PUBLIC_CHANNELS_UNKNOWN_MESSAGE not in _texts(slack)


@pytest.mark.asyncio
async def test_a_channel_question_nothing_answers_is_told_to_the_asker_alone(slack):
    """A miss is not posted to the channel."""
    slack.credential = _credential(synced=["C1"])
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("what is the german capital?"))

    assert slack.posted == []
    assert _texts(slack)[-1] == mention.NOTHING_FOUND_MESSAGE


@pytest.mark.asyncio
async def test_an_answer_slack_does_not_take_is_told_to_the_asker(slack):
    """A refused or failed ``chat.postMessage`` does not end the question silently."""
    slack.credential = _credential(synced=["C1"])
    with (
        patch.object(mention, "post_message", AsyncMock(side_effect=RuntimeError("ratelimited"))),
        patch.object(mention, "cognee_search", AsyncMock(return_value=_FOUND)),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("why did we choose Neon?"))

    assert _texts(slack)[-1] == mention.ANSWER_NOT_POSTED_MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("thread", [None, "2.0"], ids=["a new message", "in its thread"])
async def test_a_direct_message_is_answered_with_a_message_from_the_members_memory(slack, thread):
    """Goran, 2026-11-12: a direct message is answered with a real message in it, which
    stays between the member and the app and is still there after a reload. Its scope is
    today's: the workspace dataset within the selected channels Slack says the member is
    in, and the member's own datasets whole."""
    slack.credential = _credential(synced=["C1", "G2", "C3"])
    where = {"thread_ts": thread} if thread else {}
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=_FOUND)) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _direct("why did we choose Neon?") | where)

    assert _scopes(search) == [([slack.workspace.id], ["slack:channel:C1"]), ([slack.own.id], None)]
    assert len(slack.posted) == 1, "one answer, posted in the direct message"
    ((channel, thread_ts, answer),) = slack.posted
    assert (channel, thread_ts) == ("D1", thread)
    assert "branching" in answer["text"]
    assert not [p for _, _, p in slack.replies if "branching" in p["text"]], "not twice"


class _SlackApi:
    """``client_session`` standing in for the Slack Web API: records each call, answers
    ``answer``."""

    def __init__(self, answer: dict):
        self.answer, self.calls = answer, []

    def __call__(self, provider, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None, allow_redirects=True):
        self.calls.append((url, json, allow_redirects))
        return self

    status = 200

    async def json(self):
        return self.answer


@pytest.mark.asyncio
async def test_post_message_posts_in_the_thread_for_everyone(monkeypatch):
    api = _SlackApi({"ok": True})
    monkeypatch.setattr(mention, "client_session", api)
    monkeypatch.setattr(mention, "access_token_for", AsyncMock(return_value="xoxb"))

    await mention.post_message(_credential(), "C1", {"text": "hi", "blocks": [{}]}, "1.1")

    assert api.calls == [
        (
            "https://slack.com/api/chat.postMessage",
            {
                "channel": "C1",
                "text": "hi",
                "unfurl_links": False,
                "unfurl_media": False,
                "blocks": [{}],
                "thread_ts": "1.1",
            },
            False,
        )
    ]


@pytest.mark.asyncio
async def test_post_message_raises_when_slack_refuses_it(monkeypatch):
    """The caller tells the asker; an answer Slack refused is never taken as posted."""
    monkeypatch.setattr(mention, "client_session", _SlackApi({"ok": False, "error": "x"}))
    monkeypatch.setattr(mention, "access_token_for", AsyncMock(return_value="xoxb"))

    with pytest.raises(RuntimeError, match="chat.postMessage failed: x"):
        await mention.post_message(_credential(), "C1", {"text": "hi"})


@pytest.mark.asyncio
async def test_a_posted_answer_does_not_unfold_its_sources(monkeypatch):
    """Live, 2026-11-12 (Luca: "Can we kill the previews?"): every source link unfolded into a
    preview of the whole message, three cards under a one-line answer. The sources stay
    links."""
    api = _SlackApi({"ok": True})
    monkeypatch.setattr(mention, "client_session", api)
    monkeypatch.setattr(mention, "access_token_for", AsyncMock(return_value="xoxb"))

    await mention.post_message(_credential(), "C1", {"text": "hi"}, "1.1")

    ((_, body, _),) = api.calls
    assert body["unfurl_links"] is False and body["unfurl_media"] is False


# -- The answer's design and rating (SDK-1502): red until fixed ---------------------------


def _context_texts(answer: dict) -> list[str]:
    return [
        element["text"]
        for block in answer["blocks"]
        if block["type"] == "context"
        for element in block["elements"]
    ]


async def _answered(slack, event, record="qa-1"):
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=_FOUND)),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
        patch.object(mention, "record_answer", AsyncMock(return_value=record), create=True),
    ):
        await mention.handle_cognee_mention(TEAM, event)
    ((_, _, answer),) = slack.posted
    return answer


def _rating_buttons(answer: dict) -> list[tuple[str, str, str]]:
    return [
        (element["text"]["text"], element["action_id"], element["value"])
        for block in _actions(answer)
        for element in block["elements"]
    ]


@pytest.mark.asyncio
async def test_a_channel_answer_says_where_it_comes_from_and_can_be_rated(slack):
    """The scope line names this channel and counts the public channels it also read; 👍
    and 👎 carry the answer's id and nothing else."""
    slack.credential = _credential(synced=["C1", "C3", "C5"])
    slack.public = {"C1", "C3", "C5"}

    answer = await _answered(slack, _mention("why did we choose Neon?"))

    assert "From <#C1> and 2 public channels" in _context_texts(answer)
    assert _rating_buttons(answer) == [
        ("👍", "cognee_answer_up", "qa-1"),
        ("👎", "cognee_answer_down", "qa-1"),
    ]


@pytest.mark.asyncio
async def test_a_slack_connect_answer_says_it_read_this_channel_alone(slack):
    slack.credential = _credential(synced=["C1", "C3"])
    slack.external = True

    answer = await _answered(slack, _mention("why did we choose Neon?"))

    assert "From <#C1>" in _context_texts(answer)


@pytest.mark.asyncio
async def test_a_direct_message_answer_says_it_read_your_channels_and_memory(slack):
    slack.credential = _credential(synced=["C1"])

    answer = await _answered(slack, _direct("why did we choose Neon?"))

    assert "From your channels and your memory" in _context_texts(answer)
    assert [action for _, action, _ in _rating_buttons(answer)] == [
        "cognee_answer_up",
        "cognee_answer_down",
    ]


@pytest.mark.asyncio
async def test_an_answer_that_cannot_be_kept_has_no_rating_buttons(slack):
    """With the session cache off (``CACHING=false``) nothing can carry a rating, so the
    answer is posted without buttons rather than with buttons that do nothing."""
    slack.credential = _credential(synced=["C1"])

    answer = await _answered(slack, _mention("why did we choose Neon?"), record=None)

    assert "branching" in answer["text"]
    assert not _actions(answer)


@pytest.mark.asyncio
async def test_a_direct_message_answer_is_recorded_without_its_text(slack):
    """Goran, 2026-11-12: the record is kept under the workspace's owner, and a direct
    message's answer may quote the member's own datasets, so only the graph elements it used
    are kept (nothing to embed); a channel answer, public already, keeps its text."""
    slack.credential = _credential(synced=["C1"])
    record = AsyncMock(return_value="qa-1")
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=_FOUND)),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
        patch.object(mention, "record_answer", record),
    ):
        await mention.handle_cognee_mention(TEAM, _direct("why did we choose Neon?"))
        await mention.handle_cognee_mention(TEAM, _mention("why did we choose Neon?"))

    direct, channel = record.await_args_list
    assert (direct.kwargs["question"], direct.kwargs["answer"]) == ("", "")
    assert channel.kwargs["question"] == "why did we choose Neon?"
    assert "branching" in channel.kwargs["answer"]


# -- Help layout (Goran, 2026-11-12): red until fixed ------------------------------------


@pytest.mark.asyncio
async def test_help_is_laid_out_in_groups_with_a_private_footer(slack):
    await mention.handle_cognee_mention(TEAM, _mention("help"))

    ((_, _, reply),) = slack.replies
    blocks = reply["blocks"]
    texts = json.dumps(blocks)
    assert blocks[0]["text"]["text"].startswith("*cognee in Slack*")
    assert "Ask, remember and forget from any channel cognee is in." in blocks[0]["text"]["text"]
    assert [block["type"] for block in blocks].count("divider") == 3
    for group in ("*Ask*", "*Remember*", "*Forget*", "*Account*"):
        assert group in texts
    assert any(block.get("fields") for block in blocks), "label and value in two columns"
    for command in ("`@cognee <question>`", "`@cognee remember <text>`", "`@cognee status`"):
        assert command in texts
    assert blocks[-1] == {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "Only you can see this."}],
    }
    # The notification text stays readable on its own.
    assert reply["text"].startswith("cognee in Slack")
    assert "@cognee remember <text>" in reply["text"]


@pytest.mark.asyncio
async def test_status_is_answered_with_the_status_rows(slack):
    """``@cognee status`` posts ``status.status_message`` for this channel, privately."""
    message = {"text": "cognee status", "blocks": [{"type": "section", "fields": []}]}
    answer = AsyncMock(return_value=message)
    with patch.object(mention, "status_message", answer):
        await mention.handle_cognee_mention(TEAM, _mention("status", channel="C7"))

    ((channel, user, reply),) = slack.replies
    assert (channel, user) == ("C7", MEMBER)
    assert reply["blocks"] == message["blocks"]
    answer.assert_awaited_once()
    assert answer.await_args.kwargs["channel"] == "C7"
