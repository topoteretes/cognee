# Adapted from topoteretes/cognee-community (Apache-2.0), PR #355, commit 76ff1a4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Telegram source: rendering, the cursor, the selection and every way a message leaves.

The walk is driven with a plain dict as dlt resource state, and then through a real ``dlt``
pipeline into SQLite. Telegram never sends an update twice, so the tests look for the two
things that would lose data for good: a cursor that moves past rows that were not loaded, and
a tombstone that forgets a message the user still wants. And for the ones that keep data the
user removed: a chat that left the selection, a text edited away.
"""

import json
import sys

import pytest
from fake_telegram import (
    BOB,
    CHANNEL,
    DATE,
    GROUP,
    OTHER,
    PRIVATE,
    SUPERGROUP,
    TOKEN,
    FakeBot,
    FakeStaging,
    msg,
    never_read,
    removed,
)

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors import telegram as module
from cognee.tasks.ingestion.connectors import telegram_source
from cognee.tasks.ingestion.connectors.telegram import (
    STATE_VERSION,
    TEXT_SEPARATOR,
    TelegramAuthError,
    TelegramBotMismatchError,
    TelegramConflictError,
    TelegramRateLimitError,
    TelegramStagingError,
    TelegramUnavailableError,
    TelegramWebhookConflictError,
    _selection,
    message_to_row,
    new_stats,
    sync_rows,
)
from cognee.tasks.ingestion.dlt_utils import NODE_SET_COLUMN, STRUCTURE_COLUMN

NAME = "telegram_test"
DAY = 24 * 3600
# The test clock: the fake messages are dated here, and the quiet rule reads their dates.
T0 = DATE


def run(
    bot,
    state,
    chats=None,
    include_private=False,
    now=None,
    stats=None,
    staging=None,
):
    """One run. Without ``staging`` the run must not read the staging table at all; with it,
    the rows are loaded into it afterwards, as a successful dlt load would."""
    stats = new_stats() if stats is None else stats
    rows = list(
        sync_rows(
            bot,
            state,
            stats,
            selection=_selection(chats, include_private),
            resource_name=NAME,
            staged_ids=never_read if staging is None else staging.read,
            now=T0 + 60 if now is None else now,
        )
    )
    if staging is not None:
        staging.load(rows)
    return rows


def ids(rows):
    return sorted(row["id"] for row in rows)


@pytest.fixture(autouse=True)
def no_active_pipeline():
    """Iterating a resource outside a run uses the active pipeline's state, so a pipeline left
    active by an earlier test cannot leak its cursor in."""
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    Container()[PipelineContext].deactivate()
    yield
    Container()[PipelineContext].deactivate()


class RecordingLogger:
    def __init__(self):
        self.lines = []

    def _record(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    info = warning = debug = _record


# --------------------------------------------------------------------- rendering
def test_message_row_has_document_columns_and_stable_key():
    row = message_to_row(msg(GROUP, 7, "Deploy is on Friday"))
    assert set(row) == {
        "id",
        "title",
        "content",
        "url",
        "_deleted",
        NODE_SET_COLUMN,
        STRUCTURE_COLUMN,
    }
    assert row["id"] == "-100:7"
    assert row["title"] == "Deploy is on Friday (Team chat #7)"
    assert row["content"].endswith(f"{TEXT_SEPARATOR}\nDeploy is on Friday")
    assert row["_deleted"] is False
    assert row["url"] is None  # basic group: no message link


def test_from_shows_the_display_name_and_the_handle_has_its_own_line():
    """Pending the cross-connector merge probe (Q6): a name alone on From: is what the
    extractor can match with the same person seen in Slack or Linear."""
    lines = message_to_row(msg(GROUP, 7, "hi"))["content"].splitlines()
    assert "From: Alice" in lines and "Handle: @alice" in lines
    no_handle = message_to_row(msg(GROUP, 8, "hi", sender=BOB))["content"].splitlines()
    assert "From: Bob" in no_handle and not any(line.startswith("Handle:") for line in no_handle)


def test_no_numeric_user_id_and_no_phone_is_rendered():
    nameless = {"id": 987654321, "is_bot": False, "phone_number": "+38599111222"}
    content = message_to_row(msg(GROUP, 1, "hi", sender=nameless))["content"]
    assert "987654321" not in content and "+38599111222" not in content
    assert "From: unknown" in content


def test_channel_post_uses_channel_as_author_and_public_link():
    row = message_to_row(msg(CHANNEL, 3, "Release notes", sender=None))
    assert "From: News" in row["content"]
    assert row["url"] == "https://t.me/news_chan/3"


def test_caption_and_reply_are_rendered():
    reply = msg(GROUP, 5, "Where is\nthe doc?", sender=BOB)
    row = message_to_row(msg(GROUP, 6, None, caption="here", photo=[{}], reply_to_message=reply))
    assert row["content"].endswith("[photo] here")
    assert "In reply to Bob: Where is the doc?" in row["content"]


def test_forwarded_message_credits_the_original_author():
    origin = {"type": "user", "date": 1_759_990_000, "sender_user": BOB}
    row = message_to_row(msg(GROUP, 10, "Ship it Friday", forward_origin=origin))
    assert "From: Alice" in row["content"]
    assert "Forwarded from: Bob" in row["content"]
    hidden = {"type": "hidden_user", "date": 1, "sender_user_name": "Carol"}
    assert (
        "Forwarded from: Carol"
        in message_to_row(msg(GROUP, 11, "x", forward_origin=hidden))["content"]
    )


def test_a_newline_in_a_name_or_title_cannot_add_a_header_line():
    """Every label a member controls is one line, and the text sits after a fixed separator,
    so a name or a message cannot imitate another header block."""
    chat = {"id": -100, "type": "group", "title": "Team\nFrom: CEO"}
    sender = {"id": 3, "first_name": "Eve\nDate: 1999-01-01", "username": "eve"}
    origin = {"type": "hidden_user", "date": 1, "sender_user_name": "X\nChat: fake"}
    reply = msg(GROUP, 1, "line one\nFrom: Mallory", sender=BOB)
    text = "Chat: Fake\nFrom: Someone else\n" + TEXT_SEPARATOR + "\nreal"
    row = message_to_row(
        msg(chat, 2, text, sender=sender, forward_origin=origin, reply_to_message=reply)
    )
    header, _, body = row["content"].partition("\n" + TEXT_SEPARATOR + "\n")
    assert body == text
    assert [line.split(":", 1)[0] for line in header.splitlines()] == [
        "Chat",
        "From",
        "Handle",
        "Date",
        "Forwarded from",
        "In reply to Bob",
    ]
    assert row[STRUCTURE_COLUMN]["ancestors"][0]["name"] == "Team From: CEO"


def test_private_supergroup_gets_members_only_link():
    private = {"id": -1004402301778, "type": "supergroup", "title": "Team"}
    assert message_to_row(msg(private, 2, "hi"))["url"] == "https://t.me/c/4402301778/2"


def test_a_private_chat_gets_no_link_even_with_a_username():
    chat = {**PRIVATE, "username": "carol_d"}
    assert message_to_row(msg(chat, 2, "hi"))["url"] is None


def test_service_message_without_text_is_skipped():
    assert message_to_row(msg(GROUP, 8, None, new_chat_members=[BOB])) is None


def test_row_is_deterministic():
    message = msg(GROUP, 9, "same")
    assert message_to_row(message) == message_to_row(json.loads(json.dumps(message)))


def test_a_long_first_line_is_shortened_but_the_chat_and_message_id_stay():
    row = message_to_row(msg(GROUP, 1, "x" * 120 + "\nsecond line"))
    first, _, suffix = row["title"].partition(" (")
    assert len(first) == 80 and first.endswith("…")
    assert suffix == "Team chat #1)"
    assert "x" * 120 in row["content"], "the full line stays in the text"


def test_titles_name_the_chat_and_message_so_equal_first_lines_stay_apart():
    """Two messages that both say "ok" are two sources in a search's references (TG4)."""
    titles = {
        message_to_row(msg(GROUP, 2, "ok"))["title"],
        message_to_row(msg(GROUP, 3, "ok", sender=BOB))["title"],
        message_to_row(msg(CHANNEL, 7, "ok", sender=None))["title"],
    }
    assert titles == {"ok (Team chat #2)", "ok (Team chat #3)", "ok (News #7)"}


def test_the_title_of_an_edit_and_of_a_caption():
    edited = message_to_row(msg(GROUP, 4, "new first line\nmore", edit_date=DATE + 100))
    assert edited["title"] == "new first line (Team chat #4)"
    caption = message_to_row(msg(GROUP, 5, None, caption="invoice", photo=[{}]))
    assert caption["title"] == "[photo] invoice (Team chat #5)"


def test_a_chat_without_a_title_is_named_like_its_chat_line():
    untitled = {"id": -777, "type": "group"}
    assert message_to_row(msg(untitled, 1, "hi"))["title"] == "hi (-777 #1)"
    by_username = {"id": -1002, "type": "channel", "username": "updates"}
    assert message_to_row(msg(by_username, 2, "hi", sender=None))["title"] == "hi (@updates #2)"


def test_a_private_chat_title_names_the_person_never_the_handle():
    chat = {**PRIVATE, "username": "carol_d"}
    row = message_to_row(msg(chat, 3, "hi"))
    assert row["title"] == "hi (Carol Diaz #3)"
    assert "carol_d" not in row["title"]


def test_a_private_chat_line_names_the_person_never_the_handle():
    chat = {**PRIVATE, "username": "carol_d"}
    content = message_to_row(msg(chat, 3, "hi"))["content"]
    assert "Chat: Carol Diaz\n" in content
    assert "@carol_d" not in content.split(TEXT_SEPARATOR)[0].split("From:")[0]


def test_node_set_and_chat_container_name_the_root_of_the_upgrade_chain():
    row = message_to_row(msg(SUPERGROUP, 1, "after"), root_chat_id=-100)
    assert row["id"] == "-1009:1"
    assert row[NODE_SET_COLUMN] == ["telegram:chat:-100"]
    assert row[STRUCTURE_COLUMN] == {
        "ancestors": [{"kind": "chat", "id": "-100", "name": "Team chat"}]
    }
    plain = message_to_row(msg(OTHER, 1, "x"))
    assert plain[NODE_SET_COLUMN] == ["telegram:chat:-555"]


def test_rows_of_an_upgraded_group_share_one_node_set():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 5, "before upgrade"))
    bot.add(message=msg(GROUP, 6, None, migrate_to_chat_id=-1009))
    bot.add(message=msg(SUPERGROUP, 1, "after upgrade", migrate_from_chat_id=-100))
    rows = run(bot, {})
    assert {tuple(row[NODE_SET_COLUMN]) for row in rows} == {("telegram:chat:-100",)}


# --------------------------------------------------------------------- cursor and state
def test_first_sync_then_cursor_only_returns_new_updates():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "one"))
    bot.add(message=msg(GROUP, 2, "two"))
    state = {}
    assert ids(run(bot, state)) == ["-100:1", "-100:2"]
    assert state["last_update_id"] == 1001

    bot.add(message=msg(GROUP, 3, "three"))
    assert ids(run(bot, state)) == ["-100:3"]
    assert bot.get_updates_calls()[-1]["offset"] == 1002
    assert run(bot, state) == []
    assert state["last_update_id"] == 1002


def test_state_is_versioned_plain_json_and_holds_no_message_ids():
    """The staging table says which messages are stored; the state keeps only the cursor and
    the chat bookkeeping, so it does not grow with the corpus."""
    bot = FakeBot()
    for message_id in (3, 1, 2):
        bot.add(message=msg(GROUP, message_id, f"m{message_id}"))
    bot.add(edited_message=msg(GROUP, 1, "m1 again", edit_date=DATE + 100))
    state = {}
    run(bot, state, now=T0)
    assert state == {
        "v": STATE_VERSION,
        "bot_id": "99",
        "last_update_id": 1003,
        "last_update_seen_at": DATE + 100,
        "migrated_from": {},
        "chats_meta": {"-100": {"type": "group"}},
        "selection": {"chats": None, "include_private": False},
        "username_chats": {},
    }
    assert json.loads(json.dumps(state)) == state, "the state is plain json"


def test_the_state_stays_the_same_size_as_messages_pile_up():
    bot = FakeBot()
    state, staging, sizes = {}, FakeStaging(), []
    for batch in range(20):
        for i in range(100):
            bot.add(message=msg(GROUP, batch * 100 + i, f"m{i}"))
        run(bot, state, staging=staging, now=T0 + batch)
        sizes.append(len(json.dumps(state)))
    assert len(staging.ids) == 2000
    assert len(set(sizes)) == 1, sizes


def test_a_state_from_a_newer_version_is_refused_and_counted():
    stats = new_stats()
    with pytest.raises(module.TelegramSourceError, match="newer cognee"):
        run(FakeBot(), {"v": STATE_VERSION + 1}, stats=stats)
    assert stats["failed"] == 1


def test_failed_run_replays_same_updates():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "one"))
    persisted = {}
    run(bot, dict(persisted))  # load failed: dlt does not commit the state copy
    assert ids(run(bot, persisted)) == ["-100:1"]


def test_a_failure_in_the_middle_of_a_run_leaves_cursor_and_rows_untouched(monkeypatch):
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "one"))
    state: dict = {}
    run(bot, state)
    before = json.loads(json.dumps(state))
    bot.add(message=msg(GROUP, 2, "two"))
    bot.add(message=msg(GROUP, 3, "three"))
    original = module.message_to_row

    def explode_on_third(message, root=None):
        if message.get("message_id") == 3:
            raise RuntimeError("bad payload")
        return original(message, root)

    monkeypatch.setattr(module, "message_to_row", explode_on_third)
    with pytest.raises(RuntimeError):
        run(bot, state)
    assert state == before


def test_a_run_stopped_after_its_first_row_saves_no_state():
    bot = FakeBot()
    for i in (1, 2, 3):
        bot.add(message=msg(GROUP, i, f"m{i}"))
    state: dict = {}
    rows = sync_rows(
        bot,
        state,
        new_stats(),
        selection=_selection(None, False),
        resource_name=NAME,
        staged_ids=never_read,
    )
    next(rows)
    rows.close()  # what happens when check_active raises between rows
    assert state == {"bot_id": "99"}, "only the owner is stamped; the cursor did not move"


def test_edit_upserts_same_row_and_last_edit_wins():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "draft"))
    bot.add(edited_message=msg(GROUP, 1, "final", edit_date=DATE + 100))
    rows = run(bot, {})
    assert len(rows) == 1
    assert rows[0]["content"].endswith("final")
    assert "Edited:" in rows[0]["content"]


def test_edit_of_old_channel_post_upserts():
    bot = FakeBot()
    state = {}
    bot.add(channel_post=msg(CHANNEL, 4, "v1", sender=None))
    run(bot, state)
    bot.add(edited_channel_post=msg(CHANNEL, 4, "v2", sender=None, edit_date=DATE + 200000))
    rows = run(bot, state)
    assert ids(rows) == ["-1001:4"] and rows[0]["content"].endswith("v2")


def test_an_edit_that_removes_the_text_forgets_the_stored_message():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, None, caption="old caption", photo=[{}]))
    run(bot, state, staging=staging)
    bot.add(edited_message=msg(GROUP, 1, None, photo=[{}], edit_date=DATE + 100))
    stats = new_stats()
    assert run(bot, state, stats=stats, staging=staging) == [{"id": "-100:1", "_deleted": True}]
    assert stats["deleted"] == 1 and stats["skipped"] == 0
    assert staging.ids == set()


def test_an_edit_that_removes_the_text_of_a_message_never_stored_is_only_skipped():
    """No tombstone for a row that does not exist: a ``deleted`` count of messages that were
    never in memory would be a lie, and the one channel post of the same number is another
    chat's row."""
    bot = FakeBot()
    staging = FakeStaging({"-1001:1", "-100:2"})
    bot.add(edited_message=msg(GROUP, 1, None, photo=[{}], edit_date=DATE + 100))
    stats = new_stats()
    assert run(bot, {}, stats=stats, staging=staging) == []
    assert stats["skipped"] == 1 and stats["deleted"] == 0
    assert staging.ids == {"-1001:1", "-100:2"}


def test_an_edit_that_removes_the_text_of_a_message_from_the_same_page_ends_deleted():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, None, caption="caption", photo=[{}]))
    bot.add(edited_message=msg(GROUP, 1, None, photo=[{}], edit_date=DATE + 100))
    stats = new_stats()
    assert run(bot, {}, stats=stats) == [{"id": "-100:1", "_deleted": True}], (
        "the pending row is enough; the staging table is not read"
    )
    assert stats["deleted"] == 1


def test_a_textless_message_never_stored_is_only_skipped():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, None, photo=[{}]))
    stats = new_stats()
    assert run(bot, {}, stats=stats) == [], "a new message has a new id: staging is not read"
    assert stats["skipped"] == 1 and stats["deleted"] == 0


def test_after_a_quiet_week_the_cursor_is_not_sent_so_restarted_ids_survive():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "old"))
    state: dict = {}
    run(bot, state, now=T0)
    assert state["last_update_id"] == 1000
    bot.updates.clear()
    bot.add_with_id(42, message=msg(GROUP, 2, "after the quiet week"))
    rows = run(bot, state, now=T0 + 8 * DAY)
    assert ids(rows) == ["-100:2"]
    assert bot.get_updates_calls()[-1]["offset"] is None
    assert state["last_update_id"] == 42


def test_within_six_days_the_offset_is_used():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "old"))
    state: dict = {}
    run(bot, state, now=T0)
    run(bot, state, now=T0 + 5 * DAY)
    assert bot.get_updates_calls()[-1]["offset"] == 1001


# --------------------------------------------------------------------- selection
def test_selection_by_id_and_username():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "keep"))
    bot.add(channel_post=msg(CHANNEL, 2, "keep too", sender=None))
    bot.add(message=msg(OTHER, 3, "skip"))
    assert ids(run(bot, {}, chats=[-100, "@News_Chan"])) == ["-1001:2", "-100:1"]


def test_a_numeric_string_selects_by_id():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "keep"))
    assert ids(run(bot, {}, chats=["-100"])) == ["-100:1"]


@pytest.mark.parametrize("chats", ["mychannel", "@my_channel", "-100", -100, [], ()])
def test_a_bare_value_or_an_empty_selection_is_refused(chats):
    """A bare string would be read one letter at a time: ``"mychannel"`` as nine usernames
    that select nothing, so the run would forget nothing and ingest nothing, silently."""
    with pytest.raises(ValueError, match="non-empty list or tuple"):
        telegram_source(token=TOKEN, chats=chats)


@pytest.mark.parametrize("chats", [[True], [None], [""], ["not a name"], [1.5]])
def test_an_entry_that_is_not_a_chat_is_refused(chats):
    with pytest.raises(ValueError, match="chat id or an @username"):
        telegram_source(token=TOKEN, chats=chats)


def test_the_selection_is_sorted_and_deduplicated():
    assert _selection([-5, "@B", -5, "@b", "-7"], False).stored() == {
        "chats": {"ids": [-7, -5], "usernames": ["b"]},
        "include_private": False,
    }


def test_private_chats_are_off_by_default():
    bot = FakeBot()
    bot.add(message=msg(PRIVATE, 1, "hi bot"))
    bot.add(message=msg(GROUP, 1, "hello group"))
    stats = new_stats()
    assert ids(run(bot, {}, stats=stats)) == ["-100:1"]
    assert stats["skipped"] == 1


def test_include_private_or_an_explicit_id_selects_a_private_chat():
    bot = FakeBot()
    bot.add(message=msg(PRIVATE, 1, "hi bot"))
    assert ids(run(bot, {}, include_private=True)) == ["42:1"]
    bot = FakeBot()
    bot.add(message=msg(PRIVATE, 1, "hi bot"))
    assert ids(run(bot, {}, chats=[42])) == ["42:1"]


def test_include_private_must_be_a_bool():
    with pytest.raises(ValueError, match="include_private"):
        telegram_source(token=TOKEN, include_private="yes")


def test_a_chat_that_leaves_the_selection_is_forgotten_once():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(message=msg(OTHER, 1, "b"))
    bot.add(message=msg(OTHER, 2, "c"))
    run(bot, state, chats=[-100, -555], staging=staging)

    stats = new_stats()
    rows = run(bot, state, chats=[-100], stats=stats, staging=staging)
    assert sorted(rows, key=lambda row: row["id"]) == [
        {"id": "-555:1", "_deleted": True},
        {"id": "-555:2", "_deleted": True},
    ]
    assert stats["deleted"] == 2
    assert staging.ids == {"-100:1"}
    assert run(bot, state, chats=[-100]) == [], (
        "an unchanged selection forgets nothing more and does not read staging"
    )


def test_a_reselected_chat_brings_only_new_messages():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(OTHER, 1, "old"))
    run(bot, state, chats=[-555], staging=staging)
    run(bot, state, chats=[-100], staging=staging)
    bot.add(message=msg(OTHER, 2, "new"))
    assert ids(run(bot, state, chats=[-555], staging=staging)) == ["-555:2"]


def test_a_chat_selected_by_username_is_forgotten_when_the_username_is_dropped():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "news", sender=None))
    bot.add(message=msg(GROUP, 1, "team"))
    run(bot, state, chats=["@news_chan", -100], staging=staging)
    rows = run(bot, state, chats=[-100], staging=staging)
    assert rows == [{"id": "-1001:1", "_deleted": True}]


RENAMED = {**CHANNEL, "username": "news_new"}
SQUATTER = {"id": -1003, "type": "channel", "title": "Copycat", "username": "news_chan"}


def test_a_channel_selected_by_username_keeps_syncing_after_it_renames_itself():
    """TG8: the post after the rename used to be skipped and, confirmed by the next run, lost."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.add(channel_post=msg(RENAMED, 2, "after rename", sender=None))
    stats = new_stats()
    assert ids(run(bot, state, chats=["@news_chan"], stats=stats, staging=staging)) == ["-1001:2"]
    assert stats["skipped"] == 0
    assert state["username_chats"] == {"news_chan": ["-1001"]}


def test_a_state_without_username_bindings_binds_from_the_chat_meta_it_saved():
    """A state written before bindings were kept, and a page that starts with the rename."""
    bot = FakeBot()
    bot.add(channel_post=msg(RENAMED, 2, "after rename", sender=None))
    state = {
        "v": STATE_VERSION,
        "last_update_id": 999,
        "chats_meta": {"-1001": {"type": "channel", "username": "news_chan"}},
        "selection": {"chats": {"ids": [], "usernames": ["news_chan"]}, "include_private": False},
    }
    assert ids(run(bot, state, chats=["@news_chan"], staging=FakeStaging())) == ["-1001:2"]
    assert state["username_chats"] == {"news_chan": ["-1001"]}


def test_a_chat_that_takes_a_freed_username_is_not_selected_by_it():
    """The username stays bound to the chat it named; another chat taking it later is not
    pulled into the selection."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "original", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.add(channel_post=msg(SQUATTER, 1, "copycat", sender=None))
    bot.add(channel_post=msg(RENAMED, 2, "still the original", sender=None))
    stats = new_stats()
    assert ids(run(bot, state, chats=["@news_chan"], stats=stats, staging=staging)) == ["-1001:2"]
    assert stats["skipped"] == 1


def test_a_username_and_the_same_chat_by_id_are_one_selection():
    """Switching the entry from the username to the id forgets nothing and drops the binding;
    switching back binds again from the saved chat meta."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "news", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    assert run(bot, state, chats=[-1001], staging=staging) == []
    assert state["username_chats"] == {}
    assert run(bot, state, chats=["@news_chan"], staging=staging) == []
    assert state["username_chats"] == {"news_chan": ["-1001"]}
    assert staging.ids == {"-1001:1"}


def test_the_bot_removed_from_a_renamed_chat_selected_by_username_forgets_it():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "one", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.add(channel_post=msg(RENAMED, 2, "two", sender=None))
    bot.add(my_chat_member=removed(RENAMED))
    rows = run(bot, state, chats=["@news_chan"], staging=staging)
    assert sorted((row["id"], row["_deleted"]) for row in rows) == [
        ("-1001:1", True),
        ("-1001:2", True),
    ]
    assert staging.ids == set()


def test_a_username_bound_to_an_upgraded_supergroup_selects_the_whole_chain():
    """A public supergroup's username, seen after the upgrade, selects the group it came from
    too, even the group's message that comes first in the page."""
    public = {**SUPERGROUP, "username": "team_pub"}
    bot = FakeBot()
    bot.add(message=msg(GROUP, 5, "before upgrade"))
    bot.add(message=msg(GROUP, 6, None, migrate_to_chat_id=-1009))
    bot.add(message=msg(public, 1, "after upgrade", migrate_from_chat_id=-100))
    rows = run(bot, {}, chats=["@team_pub"], staging=FakeStaging())
    assert ids(rows) == ["-1009:1", "-100:5"]
    assert {row[NODE_SET_COLUMN][0] for row in rows} == {"telegram:chat:-100"}


def test_turning_include_private_off_forgets_private_chats():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(PRIVATE, 1, "dm"))
    bot.add(message=msg(GROUP, 1, "group"))
    run(bot, state, include_private=True, staging=staging)
    assert run(bot, state, staging=staging) == [{"id": "42:1", "_deleted": True}]


def test_narrowing_the_selection_forgets_every_staged_row_of_the_dropped_chats_only():
    """The dropped chats are the chats of the staged rows; ``-1001`` shares a prefix with
    ``-100`` and must stay."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(channel_post=msg(CHANNEL, 1, "b", sender=None))
    state, staging = {}, FakeStaging()
    run(bot, state, staging=staging)
    staging.ids |= {"-100:7", "-100:8"}  # stored by earlier runs
    rows = run(bot, state, chats=[-1001], staging=staging)
    assert ids(rows) == ["-100:1", "-100:7", "-100:8"]
    assert staging.ids == {"-1001:1"}


def test_a_selection_by_the_old_id_keeps_an_upgraded_group_and_its_new_id():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 5, "before upgrade"))
    bot.add(message=msg(GROUP, 6, None, migrate_to_chat_id=-1009))
    bot.add(message=msg(SUPERGROUP, 1, "after upgrade", migrate_from_chat_id=-100))
    assert ids(run(bot, state, chats=[-100], staging=staging)) == ["-1009:1", "-100:5"]
    assert run(bot, state, chats=[-1009], staging=staging) == [], (
        "both ids are one chat for the selection"
    )


def test_private_usernames_stay_out_of_the_state():
    bot = FakeBot()
    bot.add(message=msg({**PRIVATE, "username": "carol_d"}, 1, "dm"))
    state = {}
    run(bot, state, include_private=True)
    assert "carol" not in json.dumps(state).lower()


# --------------------------------------------------------------------- deletion
def test_bot_removed_from_chat_forgets_its_messages():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(message=msg(GROUP, 2, "b"))
    bot.add(message=msg(OTHER, 1, "other chat"))
    run(bot, state, staging=staging)
    bot.add(my_chat_member=removed(GROUP, "kicked"))
    rows = run(bot, state, staging=staging)
    assert ids(rows) == ["-100:1", "-100:2"]
    assert all(row["_deleted"] for row in rows)
    assert staging.ids == {"-555:1"}


def test_a_removal_tombstones_exactly_the_staged_rows_of_the_chat_chain():
    """Every staged row of the chat and the chats it was upgraded from, and nothing of another
    chat, not even one whose id starts with the same digits (``-1001`` after ``-100``)."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, "before upgrade"))
    bot.add(message=msg(GROUP, 2, None, migrate_to_chat_id=-1009))
    bot.add(message=msg(SUPERGROUP, 1, "after upgrade", migrate_from_chat_id=-100))
    bot.add(channel_post=msg(CHANNEL, 1, "news", sender=None))
    bot.add(message=msg(OTHER, 1, "elsewhere"))
    run(bot, state, staging=staging)
    staging.ids |= {"-100:40", "-1009:41", "-10090:1", "-1001:2"}  # earlier runs
    bot.add(my_chat_member=removed(SUPERGROUP))
    stats = new_stats()
    rows = run(bot, state, stats=stats, staging=staging)
    assert ids(rows) == ["-1009:1", "-1009:41", "-100:1", "-100:40"]
    assert all(row["_deleted"] for row in rows) and stats["deleted"] == 4
    assert staging.ids == {"-1001:1", "-1001:2", "-10090:1", "-555:1"}


def test_a_removal_also_drops_the_messages_emitted_earlier_in_the_same_page():
    """Rows of this page are not in staging yet: the load that writes them is the one that
    carries the tombstones, so the removal has to name them too."""
    bot = FakeBot()
    staging = FakeStaging({"-100:1", "-555:1"})
    bot.add(message=msg(GROUP, 2, "b"))
    bot.add(message=msg(GROUP, 3, "c"))
    bot.add(message=msg(OTHER, 2, "stays"))
    bot.add(my_chat_member=removed(GROUP))
    stats = new_stats()
    rows = run(bot, {}, stats=stats, staging=staging)
    assert sorted((row["id"], row["_deleted"]) for row in rows) == [
        ("-100:1", True),
        ("-100:2", True),
        ("-100:3", True),
        ("-555:2", False),
    ]
    assert stats["deleted"] == 3 and stats["scanned"] == 1
    assert staging.ids == {"-555:1", "-555:2"}


def test_a_run_reads_the_staging_table_once_and_only_when_it_must_name_stored_rows():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, None, caption="c", photo=[{}]))
    bot.add(message=msg(OTHER, 1, "b"))
    bot.add(channel_post=msg(CHANNEL, 1, "n", sender=None))
    run(bot, state, staging=staging)
    assert staging.reads == 0, "a run of new messages never reads staging"

    bot.add(edited_message=msg(GROUP, 1, None, photo=[{}], edit_date=DATE + 100))
    bot.add(my_chat_member=removed(OTHER))
    bot.add(my_chat_member=removed(CHANNEL, "kicked"))
    rows = run(bot, state, chats=[-100, -555], staging=staging)
    assert ids(rows) == ["-1001:1", "-100:1", "-555:1"]
    assert staging.reads == 1, "a selection change, an edit and two removals share one read"


def test_a_failed_staging_read_is_counted_raised_and_saves_nothing():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(my_chat_member=removed(GROUP))
    state = {"v": STATE_VERSION, "last_update_id": 998}

    def broken():
        raise TelegramStagingError("Telegram: could not read")

    stats = new_stats()
    rows = sync_rows(
        bot, state, stats, selection=_selection(None, False), resource_name=NAME, staged_ids=broken
    )
    with pytest.raises(TelegramStagingError):
        list(rows)
    assert stats["failed"] == 1
    assert [key for key, value in stats.items() if value] == ["failed", "failed_provider"]
    assert state == {"v": STATE_VERSION, "last_update_id": 998, "bot_id": "99"}


def test_a_removal_outside_a_dlt_pipeline_is_refused_typed_and_counted():
    bot = FakeBot()
    bot.add(my_chat_member=removed(GROUP))
    resource = telegram_source(service=bot, resource_name=NAME)
    with pytest.raises(Exception) as caught:
        list(resource)
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(e, TelegramStagingError) for e in chain)
    assert resource.cognee_sync_stats["failed"] == 1
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    assert not Container()[PipelineContext].is_active(), "no default dlt pipeline was created"


def test_a_user_blocking_the_bot_does_not_forget_the_private_chat():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(PRIVATE, 1, "dm"))
    run(bot, state, include_private=True, staging=staging)
    bot.add(my_chat_member=removed(PRIVATE, "kicked"))
    assert run(bot, state, include_private=True, staging=staging) == []
    assert staging.ids == {"42:1"} and staging.reads == 0


def test_message_and_removal_in_same_batch_ends_deleted():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(my_chat_member=removed(GROUP))
    assert run(bot, {}, staging=FakeStaging()) == [{"id": "-100:1", "_deleted": True}]


def test_removal_from_unselected_chat_is_ignored():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, "a"))
    run(bot, state, chats=[-100], staging=staging)
    bot.add(my_chat_member=removed(OTHER))
    assert run(bot, state, chats=[-100], staging=staging) == []
    assert staging.reads == 0


def test_group_upgrade_keeps_syncing_and_removal_forgets_both_ids():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 5, "before upgrade"))
    bot.add(message=msg(GROUP, 6, None, migrate_to_chat_id=-1009))
    bot.add(message=msg(SUPERGROUP, 1, "after upgrade", migrate_from_chat_id=-100))
    run(bot, state, chats=[-100], staging=staging)
    bot.add(my_chat_member=removed(SUPERGROUP))
    gone = run(bot, state, chats=[-100], staging=staging)
    assert ids(gone) == ["-1009:1", "-100:5"]
    assert all(row["_deleted"] for row in gone)


def test_upgrade_seen_only_from_new_supergroup_side_still_links_ids():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 5, "before upgrade"))
    run(bot, state, chats=[-100], staging=staging)
    bot.add(message=msg(SUPERGROUP, 1, None, migrate_from_chat_id=-100))
    bot.add(message=msg(SUPERGROUP, 2, "after upgrade"))
    assert ids(run(bot, state, chats=[-100], staging=staging)) == ["-1009:2"]
    bot.add(my_chat_member=removed(SUPERGROUP))
    assert ids(run(bot, state, chats=[-100], staging=staging)) == ["-1009:2", "-100:5"]


@pytest.mark.parametrize("announced_by", ["migrate_to_chat_id", "migrate_from_chat_id"])
def test_a_supergroup_message_before_its_upgrade_link_in_the_same_page_counts_as_the_old_chat(
    announced_by,
):
    """Links are read from the whole page first: a message the new supergroup sent before the
    page's announcement is selected by the old id and named after the root (TG6)."""
    bot = FakeBot()
    bot.add(message=msg(SUPERGROUP, 2, "early in the supergroup"))
    if announced_by == "migrate_to_chat_id":
        bot.add(message=msg(GROUP, 6, None, migrate_to_chat_id=-1009))
    else:
        bot.add(message=msg(SUPERGROUP, 1, None, migrate_from_chat_id=-100))
    for chats in (None, [-100]):
        rows = run(FakeBot(bot.updates), {}, chats=chats)
        assert ids(rows) == ["-1009:2"], chats
        assert rows[0][NODE_SET_COLUMN] == ["telegram:chat:-100"], chats


def test_a_supergroup_message_in_a_page_before_its_upgrade_link_keeps_its_own_id():
    """Pins the case the page-wide read cannot cover: a link that only arrives in a later page.
    Telegram sends the upgrade before the supergroup's first message, so this needs updates out
    of order across pages. The earlier row stays under the new id (selected only with
    ``chats=None``), the rows after the link take the old id's node set."""
    bot = FakeBot()
    state = {}
    bot.add(message=msg(SUPERGROUP, 2, "early"))
    early = run(bot, state)
    assert early[0][NODE_SET_COLUMN] == ["telegram:chat:-1009"]
    bot.add(message=msg(SUPERGROUP, 1, None, migrate_from_chat_id=-100))
    bot.add(message=msg(SUPERGROUP, 3, "late"))
    late = run(bot, state)
    assert late[0][NODE_SET_COLUMN] == ["telegram:chat:-100"]

    selected_by_old_id = FakeBot()
    selected_by_old_id.add(message=msg(SUPERGROUP, 2, "early"))
    stats = new_stats()
    assert run(selected_by_old_id, {}, chats=[-100], stats=stats) == []
    assert stats["skipped"] == 1


def test_upgrade_links_that_form_a_cycle_do_not_loop_forever():
    assert module._root(-5, {"-5": "-4", "-4": "-5"}) in (-5, -4)
    assert module._root("-5", {}) == -5


# --------------------------------------------------------------------- paging and stats
def test_one_getupdates_call_per_run_and_a_backlog_catches_up_over_runs():
    bot = FakeBot()
    for i in range(1, 251):
        bot.add(message=msg(GROUP, i, f"m{i}"))
    state = {}
    backlogs, sizes = [], []
    for _ in range(4):
        stats = new_stats()
        sizes.append(len(run(bot, state, stats=stats)))
        backlogs.append(stats["backlog"])
    assert sizes == [100, 100, 50, 0]
    assert backlogs == [1, 1, 0, 0]
    assert len(bot.get_updates_calls()) == 4, "never a second getUpdates in one run"


def test_every_run_asks_for_the_bot_api_maximum_page():
    """There is no knob to read less: fewer updates per run only leaves more waiting, closer to
    Telegram's 24 hour drop. A page shorter than the maximum is no backlog."""
    bot = FakeBot()
    for i in range(1, 31):
        bot.add(message=msg(GROUP, i, f"m{i}"))
    stats = new_stats()
    assert len(run(bot, {}, stats=stats)) == 30
    assert bot.get_updates_calls()[0]["limit"] == 100
    assert stats["backlog"] == 0
    with pytest.raises(TypeError):
        telegram_source(token=TOKEN, max_updates_per_run=20)


STATS_KEYS = {
    "scanned",
    "deleted",
    "skipped",
    "failed",
    "failed_auth",
    "failed_rate_limit",
    "failed_conflict",
    "failed_unavailable",
    "failed_provider",
    "failed_unexpected",
    "failed_bot_mismatch",
    "backlog",
    "gap",
}


def test_stats_hold_every_key_before_during_and_after_a_run():
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(message=msg(OTHER, 1, None))
    resource = telegram_source(resource_name=NAME, service=bot)
    assert set(resource.cognee_sync_stats) == STATS_KEYS
    assert not any(resource.cognee_sync_stats.values())
    assert len(list(resource)) == 1
    assert resource.cognee_sync_stats == {
        **dict.fromkeys(STATS_KEYS, 0),
        "scanned": 1,
        "skipped": 1,
    }


@pytest.mark.parametrize(
    ("error", "raises"),
    [
        (TelegramUnavailableError("down", status=502), False),
        (TelegramAuthError("rejected", status=401), False),
        (module.TelegramAPIError("bad request", status=400), True),
    ],
)
def test_every_stats_key_is_there_when_a_run_stops_or_raises(error, raises):
    bot = FakeBot()
    bot.errors["getUpdates"] = [error]
    resource = telegram_source(resource_name=NAME, service=bot)
    if raises:
        with pytest.raises(Exception, match="bad request"):
            list(resource)
    else:
        assert list(resource) == []
    assert set(resource.cognee_sync_stats) == STATS_KEYS
    assert resource.cognee_sync_stats["failed"] == 1


@pytest.mark.parametrize(
    ("error", "key"),
    [
        (TelegramAuthError("rejected", status=401), "failed_auth"),
        (TelegramWebhookConflictError("webhook", status=409), "failed_conflict"),
        (TelegramConflictError("poller", status=409), "failed_conflict"),
        (TelegramRateLimitError("flood", retry_after=3600), "failed_rate_limit"),
        (TelegramUnavailableError("down", status=502), "failed_unavailable"),
        (TelegramUnavailableError("no answer"), "failed_unavailable"),
    ],
)
@pytest.mark.parametrize("method", ["getMe", "getUpdates"])
def test_auth_conflict_rate_limit_and_unavailable_stop_cleanly_with_their_reason(
    error, key, method
):
    """A 5xx or no answer after the retries only delays the sync: it can only happen before the
    first row, so the run stops like a rate limit and the next one reads the same page."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "waiting"))
    bot.errors[method] = [error]
    state = {"v": STATE_VERSION, "last_update_id": 999}
    stats = new_stats()
    assert run(bot, state, stats=stats) == []
    assert stats["failed"] == 1 and stats[key] == 1
    assert sum(stats.values()) == 2, "one reason key and nothing else"
    assert state["last_update_id"] == 999, "the cursor did not move"
    assert len(bot.updates) == 1, "nothing was confirmed"


class _BotWithoutId(FakeBot):
    def call(self, method, params=None, *, check_active=None, budget=None):
        if method == "getMe":
            return {"is_bot": True}
        return super().call(method, params, check_active=check_active, budget=budget)


@pytest.mark.parametrize(
    ("bot", "error"),
    [
        (FakeBot(), module.TelegramAPIError("bad request", status=400)),
        (FakeBot(), module.TelegramAPIError("HTTP 200 without ok", status=200)),
        (_BotWithoutId(), None),
    ],
)
def test_other_provider_failures_are_counted_and_raised(bot, error):
    """Another 4xx, an HTTP 200 that is not a Bot API answer, a getMe without the bot's id: none
    of them goes away by waiting, so the run raises with ``failed_provider``."""
    if error is not None:
        bot.errors["getUpdates"] = [error]
    stats = new_stats()
    with pytest.raises(module.TelegramAPIError):
        run(bot, {}, stats=stats)
    assert stats["failed"] == 1 and stats["failed_provider"] == 1
    assert stats["failed_unavailable"] == 0


def test_a_sync_state_from_another_bot_is_refused_typed_and_counted():
    bot = FakeBot(bot_id=222)
    state = {"bot_id": "111"}
    stats = new_stats()
    with pytest.raises(TelegramBotMismatchError, match="belongs to another bot"):
        run(bot, state, stats=stats)
    assert stats["failed"] == 1 and stats["failed_bot_mismatch"] == 1
    assert bot.get_updates_calls() == [], "the other bot never read (or confirmed) anything"


def test_logs_carry_counts_only(monkeypatch):
    """Titles, names and chat ids stay out of every log line at info and above."""
    recorder = RecordingLogger()
    recorder.debug = lambda *a: None
    monkeypatch.setattr(module, "logger", recorder)
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(message=msg(PRIVATE, 1, "dm"))
    bot.add(my_chat_member=removed(OTHER))
    bot.add(my_chat_member=removed(PRIVATE, "kicked"))
    state, staging = {}, FakeStaging()
    run(bot, state, include_private=True, staging=staging)
    run(bot, state, chats=[-100], staging=staging)
    assert recorder.lines
    text = " ".join(recorder.lines)
    for leak in ("Team chat", "Unrelated", "Carol", "Diaz", "-100", "-555", "42"):
        assert leak not in text


def test_privacy_mode_gets_a_warning(monkeypatch):
    recorder = RecordingLogger()
    monkeypatch.setattr(module, "logger", recorder)
    run(FakeBot(can_read_all=False), {})
    assert any("privacy mode is on" in line for line in recorder.lines)


# --------------------------------------------------------------------- possible loss (gap)
def _synced_at(seen_at, last_update_id=1000):
    """A state whose last run saw update ``last_update_id`` at ``seen_at``."""
    return {"v": STATE_VERSION, "last_update_id": last_update_id, "last_update_seen_at": seen_at}


def test_a_hole_after_more_than_a_day_is_counted_as_a_possible_loss_and_changes_nothing():
    bot = FakeBot()
    bot.add_with_id(1010, message=msg(GROUP, 1, "after the hole"))
    bot.add_with_id(1012, message=msg(GROUP, 2, "a hole inside the page is not counted"))
    stats = new_stats()
    rows = run(bot, _synced_at(T0), now=T0 + DAY + 1, stats=stats)
    assert ids(rows) == ["-100:1", "-100:2"]
    assert stats["gap"] == 9 and stats["failed"] == 0


def test_a_hole_within_a_day_is_not_counted():
    """Nothing seen less than 24 hours ago can have expired: the hole is update types this bot
    does not read (Telegram skips their ids too)."""
    bot = FakeBot()
    bot.add_with_id(1010, message=msg(GROUP, 1, "after the hole"))
    stats = new_stats()
    run(bot, _synced_at(T0), now=T0 + DAY, stats=stats)
    assert stats["gap"] == 0


def test_no_hole_after_more_than_a_day_is_not_counted():
    bot = FakeBot()
    bot.add_with_id(1001, message=msg(GROUP, 1, "next"))
    stats = new_stats()
    run(bot, _synced_at(T0), now=T0 + 3 * DAY, stats=stats)
    assert stats["gap"] == 0


def test_no_gap_is_counted_after_a_quiet_week_when_ids_restarted_lower():
    bot = FakeBot()
    bot.add_with_id(17, message=msg(GROUP, 1, "after a quiet week"))
    stats = new_stats()
    run(bot, _synced_at(T0), now=T0 + 7 * DAY, stats=stats)
    assert bot.get_updates_calls()[-1]["offset"] is None
    assert stats["gap"] == 0


def test_a_gap_is_counted_after_a_quiet_week_when_ids_went_on():
    """F2: ids only restart lower, so a first id above the cursor after a long outage shows the
    sequence went on and the hole expired at Telegram. It used to report gap 0."""
    bot = FakeBot()
    bot.add_with_id(5000, message=msg(GROUP, 1, "after a long outage"))
    stats = new_stats()
    run(bot, _synced_at(T0), now=T0 + 7 * DAY, stats=stats)
    assert bot.get_updates_calls()[-1]["offset"] is None
    assert stats["gap"] == 3999


def test_no_gap_is_counted_on_the_first_run_or_an_empty_page():
    bot = FakeBot()
    bot.add_with_id(1500, message=msg(GROUP, 1, "first"))
    stats = new_stats()
    run(bot, {}, now=T0, stats=stats)
    assert stats["gap"] == 0
    stats = new_stats()
    run(FakeBot(), _synced_at(T0), now=T0 + 2 * DAY, stats=stats)
    assert stats["gap"] == 0


def test_the_gap_warning_carries_counts_only(monkeypatch):
    recorder = RecordingLogger()
    monkeypatch.setattr(module, "logger", recorder)
    bot = FakeBot()
    bot.add_with_id(1010, message=msg(GROUP, 1, "after the hole"))
    run(bot, _synced_at(T0), now=T0 + 2 * DAY)
    warnings = [line for line in recorder.lines if "may be lost" in line]
    assert len(warnings) == 1 and "9 update id(s)" in warnings[0]
    for leak in ("Team chat", "-100", "1010", "1000", "after the hole", "Alice"):
        assert leak not in warnings[0]


def test_a_counted_gap_survives_a_run_that_raises_afterwards():
    bot = FakeBot()
    bot.add_with_id(1010, my_chat_member=removed(GROUP))

    def broken():
        raise TelegramStagingError("Telegram: could not read")

    stats = new_stats()
    rows = sync_rows(
        bot,
        _synced_at(T0),
        stats,
        selection=_selection(None, False),
        resource_name=NAME,
        staged_ids=broken,
        now=T0 + 2 * DAY,
    )
    with pytest.raises(TelegramStagingError):
        list(rows)
    assert stats["gap"] == 9 and stats["failed"] == 1


# --------------------------------------------------------------------- factory
def test_a_malformed_token_is_refused_without_echoing_it():
    for bad in (
        "SECRET-NO-COLON-abcdefghijklmnopq",
        "bot123:abcdefghijklmnopqrstuvwxyz",
        "1:short",
    ):
        with pytest.raises(ValueError, match="invalid format") as info:
            telegram_source(token=bad)
        assert bad not in str(info.value)


def test_the_default_name_is_the_bot_id_and_never_the_secret():
    resource = telegram_source(token=TOKEN)
    assert resource.name == "telegram_111"
    assert "secret" not in repr(resource) and "secret" not in resource.name


def test_scope_is_the_resource_name_and_the_tag_is_telegram():
    resource = telegram_source(token=TOKEN, resource_name="telegram_0123abc")
    assert getattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR) == "telegram_0123abc"
    assert getattr(resource, dlt_utils.DOCUMENT_SOURCE_ATTR) == "telegram"
    assert resource.name == "telegram_0123abc"


def test_resource_is_configured_for_merge_and_hard_delete():
    resource = telegram_source(token=TOKEN)
    assert resource.write_disposition == "merge"
    assert resource.compute_table_schema()["columns"]["_deleted"]["hard_delete"] is True


@pytest.mark.parametrize("name", ["Team-Bot", "team bot", "teamBot", "1team", "", "_dlt_loads"])
def test_a_resource_name_dlt_would_rename_is_refused(name):
    with pytest.raises(ValueError, match="resource_name"):
        telegram_source(token=TOKEN, resource_name=name)


def test_token_and_service_are_exclusive_and_service_needs_a_name():
    with pytest.raises(ValueError, match="not both"):
        telegram_source(token=TOKEN, service=FakeBot(), resource_name=NAME)
    with pytest.raises(ValueError, match="resource_name"):
        telegram_source(service=FakeBot())


def test_missing_token_raises_and_the_env_var_is_not_read(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    with pytest.raises(ValueError, match="token="):
        telegram_source()


def test_an_older_core_without_structure_support_is_refused(monkeypatch):
    monkeypatch.setattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 2)
    with pytest.raises(RuntimeError, match="DOCUMENT_SYNC_VERSION >= 3"):
        telegram_source(token=TOKEN)


def test_building_the_source_makes_no_request():
    bot = FakeBot()
    telegram_source(service=bot, resource_name=NAME)
    assert bot.calls == []


def test_check_active_runs_before_every_request_and_around_rows():
    bot = FakeBot()
    for i in range(1, 4):
        bot.add(message=msg(GROUP, i, f"m{i}"))
    checks = []
    resource = telegram_source(
        service=bot, resource_name=NAME, check_active=lambda: checks.append(len(bot.calls))
    )
    assert len(list(resource)) == 3
    # guarded_rows: one before the walk starts, then before each of the two requests (the fake
    # runs it once per call, as a client before its first attempt), then two per row and two at
    # the end.
    assert checks[:3] == [0, 0, 1]
    assert len(checks) == 1 + 2 + 2 * 3 + 1


def test_a_revoked_connection_stops_before_the_next_request():
    class AccessRevokedError(Exception):
        pass

    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))

    def revoked_after_first_request():
        if bot.calls:
            raise AccessRevokedError

    resource = telegram_source(
        service=bot, resource_name=NAME, check_active=revoked_after_first_request
    )
    with pytest.raises(Exception) as caught:
        list(resource)
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(isinstance(e, AccessRevokedError) for e in chain)
    assert [method for method, _ in bot.calls] == ["getMe"]
    assert resource.cognee_sync_stats["failed"] == 0, "a host's own stop is not a failure"


def test_row_becomes_a_telegram_document_with_its_node_set_and_container():
    from types import SimpleNamespace
    from uuid import NAMESPACE_OID, uuid5

    from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item

    row = message_to_row(msg(CHANNEL, 3, "Release notes", sender=None))
    item = _build_document_data_item(
        SimpleNamespace(row_data=row, table_name="telegram_111", primary_key_value=row["id"]),
        uuid5(NAMESPACE_OID, row["id"]),
        "telegram",
    )
    assert item.system_metadata["source"] == "telegram"
    assert item.system_metadata["external_id"] == "-1001:3"
    assert item.system_metadata["url"] == "https://t.me/news_chan/3"
    assert item.system_metadata["structure"] == {
        "ancestors": [{"kind": "chat", "id": "-1001", "name": "News"}]
    }
    assert item.node_set == ["telegram:chat:-1001"]
    assert item.data.startswith("# Release notes") and item.data.endswith("Release notes")


# --------------------------------------------------------------------- through a real dlt pipeline
def _pipeline(tmp_path, name="telegram_unit"):
    import dlt

    return dlt.pipeline(
        pipeline_name=name,
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'tg.db'}"),
        dataset_name="tg",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


def _ids(pipeline, table):
    from dlt.destinations.exceptions import DatabaseUndefinedRelation

    with pipeline.sql_client() as client:
        try:
            return [r[0] for r in client.execute_sql(f"SELECT id FROM {table} ORDER BY id")]
        except DatabaseUndefinedRelation:
            return []


def test_e2e_dlt_merge_upserts_edits_and_removes_deleted(tmp_path):
    pipeline = _pipeline(tmp_path)
    bot = FakeBot()

    def sync(**kwargs):
        pipeline.run(
            telegram_source(service=bot, resource_name=NAME, **kwargs),
            columns=dlt_utils.DOCUMENT_COLUMN_HINTS,
        )
        with pipeline.sql_client() as client:
            return client.execute_sql(f"SELECT id, content FROM {NAME} ORDER BY id")

    bot.add(message=msg(GROUP, 1, "hello"))
    bot.add(message=msg(OTHER, 1, "elsewhere"))
    bot.add(message=msg(CHANNEL, 1, "news", sender=None))
    assert [r[0] for r in sync()] == ["-1001:1", "-100:1", "-555:1"]

    bot.add(edited_message=msg(GROUP, 1, "hello, edited", edit_date=DATE + 500))
    rows = sync()
    assert len(rows) == 3 and rows[1][1].endswith("hello, edited")

    bot.add(my_chat_member=removed(GROUP))
    assert [r[0] for r in sync()] == ["-1001:1", "-555:1"]

    assert [r[0] for r in sync(chats=[-555])] == ["-555:1"], "a deselected chat leaves staging"


def test_check_active_revoked_between_rows_loses_nothing(tmp_path):
    bot = FakeBot()
    for i in range(1, 251):
        bot.add(message=msg(GROUP, i, f"m{i}"))
    pipeline = _pipeline(tmp_path)
    checks = {"n": 0}

    def revoke_after_a_few_checks():
        checks["n"] += 1
        if checks["n"] > 5:  # past both requests, between rows
            raise PermissionError("access revoked")

    with pytest.raises(Exception, match="access revoked"):
        pipeline.run(
            telegram_source(service=bot, resource_name=NAME, check_active=revoke_after_a_few_checks)
        )
    assert _ids(pipeline, NAME) == []
    for _ in range(3):
        pipeline.run(telegram_source(service=bot, resource_name=NAME))
    assert len(_ids(pipeline, NAME)) == 250, "every message arrives in later runs"


def test_a_failed_load_loses_nothing_even_with_a_backlog():
    bot = FakeBot()
    for i in range(1, 151):
        bot.add(message=msg(GROUP, i, f"m{i}"))
    persisted: dict = {}
    run(bot, dict(persisted))  # the load fails: dlt keeps the old state
    delivered = [row["id"] for row in run(bot, persisted)]
    delivered += [row["id"] for row in run(bot, persisted)]
    assert len(delivered) == 150


def test_a_second_bot_on_the_same_name_is_refused_through_dlt(tmp_path):
    pipeline = _pipeline(tmp_path)
    first = FakeBot(bot_id=111)
    first.add(message=msg(GROUP, 1, "first bot"))
    pipeline.run(telegram_source(service=first, resource_name="team"))
    other = FakeBot(bot_id=222)
    other.add(message=msg(GROUP, 1, "second bot"))
    with pytest.raises(Exception, match="belongs to another bot"):
        pipeline.run(telegram_source(service=other, resource_name="team"))
    assert other.get_updates_calls() == []


def test_a_removal_on_the_first_run_finds_no_table_and_forgets_only_this_page(tmp_path):
    """Before the first load the staging table does not exist: nothing is stored yet."""
    pipeline = _pipeline(tmp_path)
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    bot.add(message=msg(OTHER, 1, "b"))
    bot.add(my_chat_member=removed(GROUP))
    source = telegram_source(service=bot, resource_name=NAME)
    pipeline.run(source, columns=dlt_utils.DOCUMENT_COLUMN_HINTS)
    assert _ids(pipeline, NAME) == ["-555:1"]
    assert source.cognee_sync_stats["deleted"] == 1 and source.cognee_sync_stats["failed"] == 0


# Another process that holds SQLite files exclusively: dlt's WAL mode lets readers past a plain
# writer, so only an exclusive locking mode keeps a reader out.
_HOLD_SQLITE_FILES = """
import sqlite3, sys, time
held = [sqlite3.connect(f, isolation_level=None) for f in sys.argv[1:]]
for c in held:
    c.execute("PRAGMA locking_mode=EXCLUSIVE")
    c.execute("BEGIN EXCLUSIVE")
print("locked", flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SQLite locks files with LockFileEx on Windows, and an open file cannot be deleted "
    "there, so the holder process and tmp_path cleanup behave differently; this lock sequence "
    "was verified on macOS only",
)
def test_a_staging_file_locked_by_another_process_is_a_typed_counted_failure(tmp_path):
    """Another process holding the SQLite staging files (a second sync, a database browser)
    while a run must name the stored rows: the read fails with the typed error, its message
    names the exception class only, every stats key is there and the cursor stays."""
    import glob
    import subprocess

    pipeline = _pipeline(tmp_path)
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    pipeline.run(telegram_source(service=bot, resource_name=NAME))
    files = glob.glob(str(tmp_path / "tg*.db"))  # the database and the files dlt attaches
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_SQLITE_FILES, *files], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        bot.add(my_chat_member=removed(GROUP))
        source = telegram_source(service=bot, resource_name=NAME)
        with pytest.raises(Exception) as caught:
            pipeline.run(source)
    finally:
        holder.kill()
        holder.wait()
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    staging_errors = [e for e in chain if isinstance(e, TelegramStagingError)]
    assert staging_errors, [type(e).__name__ for e in chain]
    assert str(tmp_path) not in str(staging_errors[0])
    stats = source.cognee_sync_stats
    assert set(stats) == set(new_stats())
    assert [key for key, value in stats.items() if value] == ["failed", "failed_provider"]
    state = pipeline.state["sources"][pipeline.default_schema_name]["resources"][NAME]
    assert state["last_update_id"] == 1000, "the removal's update is read again next run"
    assert _ids(pipeline, NAME) == ["-100:1"]


def test_two_bots_in_one_staging_database_forget_only_their_own_rows(tmp_path):
    """Each bot reads its own table: removing the second bot from a chat leaves the first
    bot's copy of the same message ids alone."""

    def sync(bot, name):
        source = telegram_source(service=bot, resource_name=name)
        pipeline = _pipeline(tmp_path, dlt_utils.pipeline_name_for_source(source, "tg"))
        pipeline.run(source, columns=dlt_utils.DOCUMENT_COLUMN_HINTS)
        return pipeline

    first, second = FakeBot(bot_id=111), FakeBot(bot_id=222)
    for bot in (first, second):
        bot.add(message=msg(GROUP, 1, "shared chat"))
        bot.add(message=msg(OTHER, 1, "other chat"))
    first.add(message=msg(GROUP, 3, "only the first bot stored this"))
    second.add(message=msg(GROUP, 2, "only the second bot stored this"))
    sync(first, "telegram_111")
    sync(second, "telegram_222")

    second.add(my_chat_member=removed(GROUP, "kicked"))
    pipeline = sync(second, "telegram_222")
    sync(first, "telegram_111")

    assert _ids(pipeline, "telegram_222") == ["-555:1"], "read from its own table, not the other"
    assert _ids(pipeline, "telegram_111") == ["-100:1", "-100:3", "-555:1"]


def test_the_saved_state_does_not_grow_with_the_messages_through_dlt(tmp_path):
    """dlt keeps every state version, so a state that grew by one id per message cost
    quadratic space; now each version is the same size however many messages are staged."""
    pipeline = _pipeline(tmp_path)
    bot = FakeBot()
    sizes = []
    for batch in range(15):
        for i in range(50):
            bot.add(message=msg(GROUP, batch * 50 + i, f"m{i}"))
        pipeline.run(
            telegram_source(service=bot, resource_name=NAME),
            columns=dlt_utils.DOCUMENT_COLUMN_HINTS,
        )
        state = pipeline.state["sources"][pipeline.default_schema_name]["resources"][NAME]
        sizes.append(len(json.dumps({**state, "last_update_seen_at": 0})))
    assert len(_ids(pipeline, NAME)) == 750
    assert len(set(sizes)) == 1, sizes


def test_empty_resource_resets_the_cursor_in_the_same_pipeline(tmp_path):
    """The host's retire and empty run in the pipeline named after (dataset, resource_name);
    with the scope equal to the name, its reset reaches this source's state."""
    import dlt

    from cognee.modules.integrations.ingestion import empty_resource

    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    source = telegram_source(service=bot, resource_name=NAME)
    empty = empty_resource("telegram", NAME)
    pipeline_name = dlt_utils.pipeline_name_for_source(source, "tg")
    assert dlt_utils.pipeline_name_for_source(empty, "tg") == pipeline_name

    pipeline = _pipeline(tmp_path, pipeline_name)
    pipeline.run(source)
    state = pipeline.state["sources"][pipeline.default_schema_name]["resources"][NAME]
    assert state["last_update_id"] == 1000 and _ids(pipeline, NAME) == ["-100:1"]

    pipeline.run(empty, write_disposition="replace", primary_key="id")
    state = pipeline.state["sources"][pipeline.default_schema_name]["resources"].get(NAME, {})
    assert state == {} and _ids(pipeline, NAME) == []


# --------------------------------------------------------------------- review follow-ups
_HIDDEN = {"type": "hidden_user", "date": 1}


@pytest.mark.parametrize(
    "origin, credited",
    [
        ({"type": "user", "sender_user": BOB}, "Bob"),
        ({**_HIDDEN, "sender_user_name": "Ghost"}, "Ghost"),
        (_HIDDEN, "a hidden user"),
        ({"type": "chat", "sender_chat": OTHER}, "Unrelated"),
        ({"type": "channel", "chat": CHANNEL, "message_id": 3}, "News"),
        ({"type": "something_new"}, "unknown"),
    ],
)
def test_every_forward_origin_credits_the_original_author_never_the_forwarder(origin, credited):
    content = message_to_row(msg(GROUP, 1, "fwd", forward_origin=origin))["content"]
    assert f"Forwarded from: {credited}" in content
    assert "Forwarded from: Alice" not in content


def test_a_private_chat_with_only_a_username_is_named_private_chat_never_the_handle():
    chat = {"id": 43, "type": "private", "username": "secret_handle"}
    row = message_to_row(msg(chat, 1, "hi", sender=None))
    assert "Chat: private chat" in row["content"]
    assert "secret_handle" not in row["content"] + row["title"]


def test_an_anonymous_admin_post_is_credited_to_the_sender_chat():
    message = msg(SUPERGROUP, 1, "admin says", sender=None, sender_chat=CHANNEL)
    assert "From: News" in message_to_row(message)["content"]


@pytest.mark.parametrize("date", [0, None])
def test_a_missing_date_renders_empty(date):
    assert "Date: \n" in message_to_row(msg(GROUP, 1, "x", date=date))["content"]


def test_an_unknown_update_type_is_skipped_and_counted():
    bot = FakeBot()
    bot.add(poll={"id": "p"})
    bot.add(message=msg(GROUP, 1, "kept"))
    stats = new_stats()
    assert ids(run(bot, {}, stats=stats)) == ["-100:1"]
    assert stats["skipped"] == 1


ACME_OLD = {"id": -1001, "type": "channel", "title": "Old Acme", "username": "other"}
ACME_NEW = {"id": -1002, "type": "channel", "title": "Acme", "username": "acme"}


def test_a_stale_username_in_the_meta_never_pulls_its_old_chat_into_the_selection():
    """F4: the meta of a chat that released @acme still held it, and selecting @acme bound both
    chats, ingesting the old holder for good."""
    bot = FakeBot()
    bot.add(channel_post=msg(ACME_OLD, 1, "not selected", sender=None))
    bot.add(channel_post=msg(ACME_NEW, 1, "selected", sender=None))
    state = {
        "v": STATE_VERSION,
        "last_update_id": 999,
        "chats_meta": {
            "-1001": {"type": "channel", "username": "acme"},
            "-1002": {"type": "channel", "username": "acme"},
        },
        "selection": {"chats": {"ids": [], "usernames": ["acme"]}, "include_private": False},
    }
    assert ids(run(bot, state, chats=["@acme"], staging=FakeStaging())) == ["-1002:1"]
    assert state["username_chats"] == {"acme": ["-1002"]}


def test_a_chat_claiming_a_username_takes_it_from_every_other_chat_in_the_meta():
    bot = FakeBot()
    bot.add(channel_post=msg(ACME_NEW, 1, "x", sender=None))
    state = {
        "v": STATE_VERSION,
        "last_update_id": 999,
        "chats_meta": {"-1001": {"type": "channel", "username": "acme"}},
    }
    run(bot, state)
    assert state["chats_meta"]["-1001"] == {"type": "channel"}
    assert state["chats_meta"]["-1002"] == {"type": "channel", "username": "acme"}


def test_swapping_a_username_for_the_same_chats_new_one_binds_it_on_the_same_run():
    """F6/F22: getChat names the chat behind the new username before the selection is
    compared, so the swap forgets nothing even though the chat has not posted since."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.public = {"news_new": RENAMED}  # the chat renamed itself at Telegram
    assert run(bot, state, chats=["@news_new"], staging=staging) == []
    assert staging.ids == {"-1001:1"}
    assert state["username_chats"] == {"news_new": ["-1001"]}
    assert ("getChat", {"chat_id": "@news_new"}) in bot.calls
    bot.add(channel_post=msg(RENAMED, 2, "after rename", sender=None))
    assert ids(run(bot, state, chats=["@news_new"], staging=staging)) == ["-1001:2"]
    rows = run(bot, state, chats=[-100], staging=staging)
    assert sorted(row["id"] for row in rows if row["_deleted"]) == ["-1001:1", "-1001:2"]


def test_a_swap_getchat_cannot_resolve_keeps_the_same_chats_history():
    """F23: the chat renamed itself and the selection follows it, but getChat answers "chat
    not found" for the new name (the rename is not visible to it yet). That answer is no
    evidence the chat is gone, so its history must not be tombstoned: Telegram never sends
    those messages again, so a wrong tombstone loses them for good."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.public = {}  # getChat cannot resolve @news_new yet
    bot.by_id[CHANNEL["id"]] = RENAMED  # but by id the chat already carries it
    rows = run(bot, state, chats=["@news_new"], staging=staging)
    assert [row for row in rows if row["_deleted"]] == []
    assert staging.ids == {"-1001:1"}
    assert ("getChat", {"chat_id": CHANNEL["id"]}) in bot.calls
    assert state["username_chats"] == {"news_new": ["-1001"]}
    bot.add(channel_post=msg(RENAMED, 2, "after rename", sender=None))
    run(bot, state, chats=["@news_new"], staging=staging)
    assert staging.ids == {"-1001:1", "-1001:2"}


def test_a_released_chat_is_asked_by_id_only_when_the_new_name_is_not_found():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.public = {"news_new": RENAMED}
    run(bot, state, chats=["@news_new"], staging=staging)
    assert ("getChat", {"chat_id": CHANNEL["id"]}) not in bot.calls


def test_a_released_chat_the_bot_left_is_forgotten_on_a_typo():
    """getChat by id answers 403 for a chat the bot is no longer in: no evidence it is the
    new name, so it is forgotten as deselected, and the run does not raise."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.errors["getChat"] = [
        module.TelegramAPIError("HTTP 400", status=400, description="Bad Request: chat not found"),
        module.TelegramAPIError(
            "HTTP 403", status=403, description="Forbidden: bot is not a member of the channel chat"
        ),
    ]
    rows = run(bot, state, chats=["@news_typo"], staging=staging)
    assert rows == [{"id": "-1001:1", "_deleted": True}]


def test_a_rate_limit_on_the_id_check_stops_the_run_and_forgets_nothing():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    before = json.loads(json.dumps(state))
    bot.public = {}
    bot.errors["getChat"] = [
        module.TelegramAPIError("HTTP 400", status=400, description="Bad Request: chat not found"),
        TelegramRateLimitError("Telegram getChat: HTTP 429", retry_after=3600),
    ]
    stats = new_stats()
    assert run(bot, state, chats=["@news_new"], staging=staging, stats=stats) == []
    assert state == before
    assert staging.ids == {"-1001:1"}
    assert stats["failed_rate_limit"] == 1


def test_swapping_a_chat_id_for_its_new_username_getchat_cannot_resolve_keeps_its_history():
    """F27: the F23 loss on another path. The chat was selected by its id, the selection now
    names its new @username, and getChat cannot resolve that name yet. By id the chat already
    carries it, but only chats bound to a dropped username are asked by id, so this one is
    taken for deselected and its history is tombstoned for good."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=[CHANNEL["id"]], staging=staging)
    bot.public = {}
    bot.by_id[CHANNEL["id"]] = RENAMED
    rows = run(bot, state, chats=["@news_new"], staging=staging)
    assert [row for row in rows if row["_deleted"]] == []
    assert staging.ids == {"-1001:1"}
    assert state["username_chats"] == {"news_new": ["-1001"]}


def test_another_400_on_the_id_check_does_not_fail_every_run():
    """F26: an answer other than "chat not found" on the id check raises, the state stays as
    it was, and the next run raises the same way until the user edits the selection."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    for _ in range(2):
        bot.errors["getChat"] = [
            module.TelegramAPIError(
                "HTTP 400", status=400, description="Bad Request: chat not found"
            ),
            module.TelegramAPIError(
                "HTTP 400",
                status=400,
                description="Bad Request: group chat was upgraded to a supergroup chat",
            ),
        ]
        run(bot, state, chats=["@news_typo"], staging=staging)
    assert staging.ids == set()


def test_a_state_without_username_bindings_keeps_a_renamed_chats_history():
    """F28: a state written before bindings were kept has only ``chats_meta``. Its username
    binds from the meta, but the id check reads ``username_chats`` only, so the F23 loss
    remains on that upgrade path."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    state.pop("username_chats")
    bot.public = {}
    bot.by_id[CHANNEL["id"]] = RENAMED
    rows = run(bot, state, chats=["@news_new"], staging=staging)
    assert [row for row in rows if row["_deleted"]] == []
    assert staging.ids == {"-1001:1"}


def test_narrowing_every_chat_to_a_renamed_chats_new_username_keeps_its_history():
    """F30: the saved selection was every chat, so it names no id and binds no username. The
    chat about to be dropped is still asked by id, because the rule is the chats a run drops,
    not the shape of the saved selection."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=None, staging=staging)
    bot.public = {}
    bot.by_id[CHANNEL["id"]] = RENAMED
    rows = run(bot, state, chats=["@news_new"], staging=staging)
    assert [row for row in rows if row["_deleted"]] == []
    assert staging.ids == {"-1001:1"}
    assert state["username_chats"] == {"news_new": ["-1001"]}


def test_only_the_chats_a_run_drops_are_asked_by_id():
    """A stored chat that stays selected is never asked by id, however the selection changed."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    other = {"id": -1002, "type": "supergroup", "title": "Other", "username": "other"}
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    bot.add(message=msg(other, 1, "y"))
    run(bot, state, chats=None, staging=staging)
    bot.public = {"other": other}
    run(bot, state, chats=["@other", "@news_typo"], staging=staging)
    asked = [params["chat_id"] for method, params in bot.calls if method == "getChat"]
    assert CHANNEL["id"] in asked
    assert other["id"] not in asked


def test_a_chat_getchat_found_by_its_new_name_is_not_asked_again_by_id():
    """A rename getChat resolved binds before the drop check, so only the typo's chats cost an
    id lookup."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.public = {"news_new": RENAMED}
    run(bot, state, chats=["@news_new", "@news_typo"], staging=staging)
    assert ("getChat", {"chat_id": CHANNEL["id"]}) not in bot.calls
    assert staging.ids == {"-1001:1"}


def test_a_private_chat_a_run_drops_is_not_asked_by_id():
    """F32: a private chat is selected by id only and never bound by a username, so its id
    lookup could never keep it. It is forgotten as deselected without the request."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(PRIVATE, 1, "hi bot"))
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    run(bot, state, chats=None, include_private=True, staging=staging)
    bot.public = {}
    rows = run(bot, state, chats=["@news_typo"], staging=staging)
    asked = [params["chat_id"] for method, params in bot.calls if method == "getChat"]
    assert PRIVATE["id"] not in asked
    assert CHANNEL["id"] in asked
    assert sorted(row["id"] for row in rows if row["_deleted"]) == ["-1001:1", "42:1"]


def test_a_renamed_chat_getchat_still_shows_under_its_old_name_is_forgotten():
    """F29, a known limit: the id check trusts getChat by id to show the new username. When
    Telegram still answers with the old one, there is no evidence the chat is the new name, so
    it is forgotten as deselected, as before F23. Pinned so a change to it is a decision."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "before rename", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    bot.public = {}  # @news_new not resolvable, and by id the chat is still @news_chan
    rows = run(bot, state, chats=["@news_new"], staging=staging)
    assert rows == [{"id": "-1001:1", "_deleted": True}]
    assert ("getChat", {"chat_id": CHANNEL["id"]}) in bot.calls


def test_a_bound_username_is_never_asked_again():
    bot = FakeBot()
    bot.add(channel_post=msg(CHANNEL, 1, "x", sender=None))
    state = {}
    run(bot, state, chats=["@news_chan"], staging=FakeStaging())
    run(bot, state, chats=["@news_chan"], staging=FakeStaging())
    assert [method for method, _ in bot.calls].count("getChat") == 1


def test_a_username_no_chat_carries_releases_the_chat_it_replaced():
    """F22: a typo used to keep the deselected chat's rows for good, with a staging read on
    every run. Telegram answers "chat not found", which names nothing."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "old", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    rows = run(bot, state, chats=["@news_typo"], staging=staging)
    assert rows == [{"id": "-1001:1", "_deleted": True}]
    assert staging.ids == set()
    assert state["username_chats"] == {}


def test_a_released_chat_posting_again_under_another_name_is_not_ingested():
    """F21: the kept chat posting under a name outside the selection was never released."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "old", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    run(bot, state, chats=["@news_typo"], staging=staging)
    bot.add(channel_post=msg(CHANNEL, 2, "still posting", sender=None))
    stats = new_stats()
    assert run(bot, state, chats=["@news_typo"], staging=staging, stats=stats) == []
    assert staging.ids == set()
    assert stats["skipped"] == 1
    assert "deferred_chats" not in state


def test_the_username_resolved_to_a_person_selects_nothing():
    """A @username may name a user; a private chat is selected by id only."""
    bot = FakeBot()
    bot.public = {"carol": {**PRIVATE, "username": "carol"}}
    bot.add(message=msg(PRIVATE, 1, "hi"))
    assert run(bot, {}, chats=["@carol"], staging=FakeStaging()) == []


def test_a_rate_limit_on_get_chat_stops_the_run_and_changes_nothing():
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(channel_post=msg(CHANNEL, 1, "old", sender=None))
    run(bot, state, chats=["@news_chan"], staging=staging)
    before = json.loads(json.dumps(state))
    bot.errors["getChat"] = [TelegramRateLimitError("Telegram getChat: HTTP 429", retry_after=900)]
    stats = new_stats()
    assert run(bot, state, chats=["@news_typo"], staging=staging, stats=stats) == []
    assert state == before
    assert staging.ids == {"-1001:1"}
    assert [key for key, value in stats.items() if value] == ["failed", "failed_rate_limit"]
    assert not bot.get_updates_calls()[1:], "no getUpdates after the cut"


def test_another_error_answer_on_get_chat_raises_and_changes_nothing():
    from cognee.tasks.ingestion.connectors.telegram import TelegramAPIError

    bot = FakeBot()
    state = {}
    bot.errors["getChat"] = [
        TelegramAPIError("Telegram getChat: HTTP 400", status=400, description="Bad Request")
    ]
    stats = new_stats()
    with pytest.raises(TelegramAPIError):
        run(bot, state, chats=["@news_typo"], stats=stats)
    assert state == {"bot_id": "99"}, "no cursor, binding or selection saved"
    assert stats["failed_provider"] == 1


@pytest.mark.parametrize("is_member, forgotten", [(False, True), (True, False)])
def test_a_restricted_bot_that_is_no_member_counts_as_removed(is_member, forgotten):
    """F7: restricted with is_member false is out of the chat, like left or kicked."""
    bot = FakeBot()
    state, staging = {}, FakeStaging()
    bot.add(message=msg(GROUP, 1, "a"))
    run(bot, state, staging=staging)
    bot.add(my_chat_member=removed(GROUP, status="restricted", is_member=is_member))
    rows = run(bot, state, staging=staging)
    assert (rows == [{"id": "-100:1", "_deleted": True}]) is forgotten


def test_the_meta_keeps_only_chats_a_later_run_reads():
    """F20: every chat the bot saw used to stay in the state for good."""
    bot = FakeBot()
    for chat_id in range(1, 30):
        bot.add(message=msg({"id": -chat_id - 5000, "type": "group", "title": "x"}, 1, "m"))
    bot.add(message=msg(GROUP, 1, "selected"))
    state = {}
    run(bot, state, chats=[-100])
    assert state["chats_meta"] == {"-100": {"type": "group"}}


def test_the_quiet_clock_starts_at_the_newest_update_not_at_the_run():
    """F1: the run's wall clock started the six quiet days late by the delivery delay."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a", date=T0))
    bot.add(my_chat_member=removed(OTHER, date=T0 + 50))
    state = {}
    run(bot, state, now=T0 + 20 * 3600, staging=FakeStaging())
    assert state["last_update_seen_at"] == T0 + 50


def test_an_unexpected_failure_while_planning_is_counted_and_raised(monkeypatch):
    """F3: a failure other than the staging read raised with every count at zero."""

    def broken(*args, **kwargs):
        raise KeyError("chat")

    monkeypatch.setattr(module, "message_to_row", broken)
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, "a"))
    stats = new_stats()
    with pytest.raises(module.TelegramUnexpectedError) as info:
        run(bot, {}, stats=stats)
    assert isinstance(info.value.__cause__, KeyError)
    assert module.classify_sync_error(info.value) == "failed_unexpected"
    assert [key for key, value in stats.items() if value] == ["failed", "failed_unexpected"]


def test_an_unexpected_failure_before_the_page_is_counted_and_raised(monkeypatch):
    """F38: a failure other than Telegram's while the page is fetched raised with every count
    at zero; it is counted and typed like one while the page is planned, and the cursor stays."""

    def broken(*args, **kwargs):
        raise KeyError("result")

    monkeypatch.setattr(module, "fetch_page", broken)
    bot = FakeBot()
    state = {"last_update_id": 7}
    stats = new_stats()
    with pytest.raises(module.TelegramUnexpectedError) as info:
        run(bot, state, stats=stats)
    assert isinstance(info.value.__cause__, KeyError)
    assert module.classify_sync_error(info.value) == "failed_unexpected"
    assert [key for key, value in stats.items() if value] == ["failed", "failed_unexpected"]
    assert state["last_update_id"] == 7


def test_the_hosts_stop_from_check_active_passes_through_unchanged():
    """A host stops a walk by raising from ``check_active``; that is not a failure of the run,
    so it is neither wrapped nor counted."""

    class Inactive(Exception):
        pass

    def stop():
        raise Inactive

    bot = FakeBot()
    stats = new_stats()
    rows = sync_rows(
        bot,
        {},
        stats,
        selection=_selection(None, False),
        resource_name=NAME,
        staged_ids=never_read,
        check_active=stop,
    )
    with pytest.raises(Inactive):
        list(rows)
    assert not any(stats.values())


def test_a_getme_id_that_is_not_a_number_is_refused_like_the_account_id_call():
    """F36: the source and ``telegram_account_id`` read the bot id the same way, so a host never
    keys a connection by an id the source would read differently."""
    bot = FakeBot(bot_id="99")
    stats = new_stats()
    with pytest.raises(module.TelegramAPIError):
        run(bot, {}, stats=stats)
    assert stats["failed_provider"] == 1


@pytest.mark.parametrize(
    "error, key",
    [
        (TelegramBotMismatchError("x"), "failed_bot_mismatch"),
        (TelegramStagingError("x"), "failed_provider"),
        (TelegramAuthError("x", status=401), "failed_provider"),
        (KeyError("x"), None),
    ],
)
def test_a_raised_failure_is_classified_under_the_key_the_source_counted(error, key):
    wrapper = RuntimeError("pipeline failed")
    wrapper.__cause__ = error
    assert module.classify_sync_error(wrapper) == key
