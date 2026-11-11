"""``@cognee``: a mention in a channel, or a direct message to the app, is the one way to
talk to Cognee from Slack (``@cognee <question>``, ``@cognee remember …``,
``@cognee link``, ``@cognee help``). Every answer is ephemeral: only the member who asked
sees it. Slack is stood in for by ``post_ephemeral`` and ``FakeSlack``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.integrations.slack import handle_cognee_mention as mention
from cognee.modules.integrations.slack import handle_slack_event as events
from cognee.modules.integrations.slack.handle_slack_link import NOT_LINKED_MESSAGE
from cognee.modules.integrations.slack.persistence import CHANNEL_NOT_ENABLED_MESSAGE
from cognee.tests.unit.tasks.connectors.fake_slack import FakeSlack

TEAM = "T0TEAM"
BOT = "U0BOT"
MEMBER = "U0MEMBER"
OWNER_ID = "11111111-1111-1111-1111-111111111111"


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
    """The connected workspace, the member's link, and every ephemeral reply."""
    state = SimpleNamespace(credential=_credential(), owner=OWNER_ID, replies=[], threads=[])

    async def post_ephemeral(credential, channel, user, payload, thread_ts=None):
        state.replies.append((channel, user, payload))
        state.threads.append(thread_ts)

    monkeypatch.setattr(
        mention, "get_by_team", AsyncMock(side_effect=lambda team: state.credential)
    )
    monkeypatch.setattr(
        mention, "resolve_owner_user_id", AsyncMock(side_effect=lambda *args: state.owner)
    )
    monkeypatch.setattr(mention, "post_ephemeral", post_ephemeral)
    return state


def _texts(state) -> list[str]:
    return [payload["text"] for _, _, payload in state.replies]


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
async def test_a_question_is_answered_privately_with_share_and_discard(slack):
    results = [{"text_result": ["We chose Neon for branching."], "evidence": []}]
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=results)) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("why did we choose Neon?"))

    assert search.await_args.kwargs["query_text"] == "why did we choose Neon?"
    # G10: the answer names the documents it read, so the search returns them.
    assert search.await_args.kwargs["include_references"] is True
    assert search.await_args.kwargs["verbose"] is True
    *_, (channel, user, answer) = slack.replies
    assert (channel, user) == ("C1", MEMBER)
    assert "Neon" in answer["text"]
    actions = [block for block in answer["blocks"] if block["type"] == "actions"]
    assert [button["text"]["text"] for button in actions[0]["elements"]] == ["Share", "Discard"]


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

    assert sent == {"channel": "C1", "user": MEMBER, "text": "hi", "thread_ts": "1.1"}


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


@pytest.mark.asyncio
async def test_a_question_nothing_answers_says_so_without_share_buttons(slack):
    results = [
        {
            "text_result": ["The capital city is not specified in the provided context."],
            "evidence": [],
        }
    ]
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=results)),
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("what is the german capital?"))

    *_, (_, _, answer) = slack.replies
    assert answer["text"] == mention.NOTHING_FOUND_MESSAGE
    assert not [block for block in answer["blocks"] if block["type"] == "actions"]


# --- unlink, status, here ---


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
async def test_status_says_privately_whether_this_channel_is_synced_and_who_you_are(slack):
    from datetime import datetime, timezone

    slack.credential.provider_metadata["selected_channels"] = ["C1"]
    slack.credential.last_synced_at = datetime(2026, 11, 9, 8, 30, tzinfo=timezone.utc)
    slack.credential.sync_status = "ok"
    with patch.object(
        mention, "get_user", AsyncMock(return_value=SimpleNamespace(email="ana@example.com"))
    ):
        await mention.handle_cognee_mention(TEAM, _mention("status"))

    ((channel, user, payload),) = slack.replies
    assert (channel, user) == ("C1", MEMBER)
    text = payload["text"]
    assert "ana@example.com" in text
    assert "synced" in text and "2026-11-09 08:30 UTC" in text


@pytest.mark.asyncio
async def test_status_of_a_member_who_never_linked_says_so(slack):
    slack.owner = None
    await mention.handle_cognee_mention(TEAM, _mention("status", channel="C2"))

    text = _texts(slack)[-1]
    assert "not linked" in text and "not synced" in text


@pytest.mark.asyncio
async def test_here_asks_within_this_channel_only(slack):
    results = [{"text_result": ["Deploys go out on Fridays."], "evidence": []}]
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=results)) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("here when do we deploy?", channel="C7"))

    assert search.await_args.kwargs["query_text"] == "when do we deploy?"
    assert search.await_args.kwargs["node_name"] == ["slack:channel:C7"]


@pytest.mark.asyncio
async def test_a_plain_question_searches_all_of_the_members_memory(slack):
    with (
        patch.object(mention, "cognee_search", AsyncMock(return_value=[])) as search,
        patch.object(mention, "get_user", AsyncMock(return_value=SimpleNamespace(id=OWNER_ID))),
    ):
        await mention.handle_cognee_mention(TEAM, _mention("when do we deploy?"))

    assert search.await_args.kwargs.get("node_name") is None


@pytest.mark.asyncio
async def test_here_in_a_direct_message_explains_that_it_needs_a_channel(slack):
    direct = {"type": "message", "channel_type": "im", "user": MEMBER, "channel": "D1", "ts": "2.2"}
    with patch.object(mention, "cognee_search", AsyncMock()) as search:
        await mention.handle_cognee_mention(TEAM, direct | {"text": "here when do we deploy?"})

    search.assert_not_awaited()
    assert _texts(slack)[-1] == mention.HERE_NEEDS_A_CHANNEL_MESSAGE
