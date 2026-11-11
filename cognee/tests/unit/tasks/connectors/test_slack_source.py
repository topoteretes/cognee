"""``slack_source``: rendering, the fingerprint re-walk, tombstones, cuts and resume.

The walker is driven with a plain dict for its state; the dlt behaviour (merge,
hard delete, state kept across runs, state rolled back on a raise) goes through a
real ``dlt.pipeline`` on SQLite in ``tmp_path``.
"""

import json
import random
from datetime import datetime, timezone

import pytest

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors import slack as slack_module
from cognee.tasks.ingestion.connectors.slack import (
    SlackAPIError,
    SlackAuthError,
    SlackMissingScopeError,
    SlackSourceError,
    SlackUnavailableError,
    _iter_rows,
    slack_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_COLUMN_HINTS
from cognee.tests.unit.tasks.connectors.fake_slack import (
    BOT_ID,
    BOT_USER,
    CHANNEL,
    TEAM,
    FakeSlack,
    missing_scope,
    rate_limited,
)

TABLE = "slack_channel_test"
STATS_KEYS = {"scanned", "emitted", "skipped", "deleted", "failed", "requests"}
# 2026-11-05 10:00:00 UTC and the minutes after it
T0 = 1793872800
DAY = 86_400
# 2026-11-05 00:00:00 UTC, the start of T0's day
MIDNIGHT = T0 - 10 * 3600


def ts(minute: int) -> str:
    return f"{T0 + minute * 60}.000100"


def _at(day: int, minute: int) -> str:
    return f"{T0 + day * DAY + minute * 60}.000100"


def _day_id(day_ts: str, channel: str = CHANNEL) -> str:
    """The id of the document of the messages outside threads of ``day_ts``'s day (G7)."""
    day = datetime.fromtimestamp(int(day_ts.split(".")[0]), timezone.utc).strftime("%Y-%m-%d")
    return f"{channel}:day:{day}"


def _six_messages() -> FakeSlack:
    """Six messages without replies, one a day (``_at(1, 0)`` to ``_at(6, 0)``), so each
    is its own day document; two to a history page."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", "ana")
    for day in range(1, 7):
        fake.post(_at(day, 0), "U1", f"message {day}")
    return fake


def _workspace() -> FakeSlack:
    fake = FakeSlack()
    fake.add_user("U1", display_name="ana", real_name="Ana Lee")
    fake.add_user("U2", real_name="Ben Ortiz")
    fake.post(ts(1), "U1", "Where is the launch checklist?")
    fake.reply(ts(1), ts(2), "U2", "In the <@U1> drive folder.")
    fake.post(ts(3), "U2", "Deploy is done.", files=[{"name": "deploy.log"}, {"title": "graph"}])
    return fake


def _stats() -> dict[str, int]:
    return dict.fromkeys(STATS_KEYS, 0)


def _walk(fake, state, stats=None, **kwargs) -> list[dict]:
    stats = stats if stats is not None else _stats()
    return list(_iter_rows(fake, TEAM, CHANNEL, state, stats, **kwargs))


def _ids(rows) -> list[str]:
    return [row["id"] for row in rows]


# --- rendering ---


def test_a_root_and_its_replies_are_one_document_with_names_and_attachments():
    rows = {row["id"]: row for row in _walk(_workspace(), {})}

    # The thread is its own document; the message without replies is its day's (G7).
    assert set(rows) == {f"{CHANNEL}:{ts(1)}", _day_id(ts(3))}
    thread = rows[f"{CHANNEL}:{ts(1)}"]
    # The title: the first words, the author and the date, so titles rarely repeat and a
    # channel rename re-cognifies nothing (G11); authors by full name (G8).
    assert thread["title"] == "Where is the launch checklist? — Ana Lee, 2026-11-05"
    assert thread["content"] == (
        "[2026-11-05 10:01 UTC] Ana Lee: Where is the launch checklist?\n\n"
        "[2026-11-05 10:02 UTC] Ben Ortiz: In the @Ana Lee drive folder."
    )
    assert thread["url"] == f"https://acme.slack.com/archives/{CHANNEL}/p{ts(1).replace('.', '')}"
    assert thread["_deleted"] is False
    assert thread[dlt_utils.NODE_SET_COLUMN] == [f"slack:channel:{CHANNEL}"]
    assert thread[dlt_utils.STRUCTURE_COLUMN] == {
        "ancestors": [{"kind": "slack_channel", "id": CHANNEL, "name": "#general"}]
    }
    day = rows[_day_id(ts(3))]
    assert day["content"] == (
        "[2026-11-05 10:03 UTC] Ben Ortiz: Deploy is done.\nAttached: deploy.log, graph"
    )
    assert day["url"] == f"https://acme.slack.com/archives/{CHANNEL}/p{ts(3).replace('.', '')}"
    # Exactly the document columns: every other column would be hashed and never read.
    assert set(thread) == {
        "id",
        "title",
        "content",
        "url",
        "_deleted",
        dlt_utils.NODE_SET_COLUMN,
        dlt_utils.STRUCTURE_COLUMN,
    }


def test_the_title_keeps_only_the_first_words_of_a_long_message():
    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(ts(1), "U1", " ".join(f"word{n}" for n in range(30)))

    (row,) = _walk(fake, {})
    assert row["title"] == (
        "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9… — Ana Lee, 2026-11-05"
    )


def test_slack_markup_reaches_the_document_as_plain_text():
    """Only the text a person reads, never Slack's link syntax or escapes (G48)."""
    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(
        ts(1),
        "U1",
        "<!here> see <#C0OPS|ops> and <https://x.example/runbook|the runbook>, "
        "<!subteam^S01|@oncall> &amp; <mailto:a@x.example|a@x.example>: 1 &lt; 2 &gt; 0 <https://y.example>",
    )

    (row,) = _walk(fake, {})
    text = row["content"].split(": ", 1)[1]
    assert text == (
        "@here see #ops and the runbook (https://x.example/runbook), "
        "@oncall & a@x.example: 1 < 2 > 0 https://y.example"
    )


def test_a_shared_message_a_bot_alert_and_a_blocks_message_keep_their_text():
    """Content that Slack carries in attachments or blocks, not in text (G49)."""
    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(
        ts(1),
        "U1",
        "",
        attachments=[{"is_share": True, "author_name": "Ben", "text": "The launch is Friday."}],
    )
    fake.post(
        ts(2),
        None,
        "",
        subtype="bot_message",
        bot_id="B9",
        username="PagerDuty",
        attachments=[
            {
                "title": "CPU high",
                "text": "node-3 at 95%",
                "fields": [{"title": "Severity", "value": "P1"}],
            }
        ],
    )
    fake.post(
        ts(3),
        "U1",
        "",
        blocks=[
            {"type": "header", "text": {"type": "plain_text", "text": "Release notes"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Deploy *v2* on Friday"}},
        ],
    )

    (row,) = _walk(fake, {})  # one day, so one document
    assert row["content"] == (
        "[2026-11-05 10:01 UTC] Ana Lee: The launch is Friday.\n\n"
        "[2026-11-05 10:02 UTC] PagerDuty: CPU high\nnode-3 at 95%\nSeverity: P1\n\n"
        "[2026-11-05 10:03 UTC] Ana Lee: Release notes\nDeploy *v2* on Friday"
    )


def test_non_content_messages_and_the_apps_own_messages_are_left_out():
    fake = _workspace()
    fake.post(ts(4), "U1", "ana has joined the channel", subtype="channel_join")
    fake.post(ts(5), BOT_USER, "Cognee answered this.")
    fake.post(ts(6), None, "Cognee via bot id", bot_id=BOT_ID, subtype="bot_message")
    fake.post(ts(7), None, "CI passed", bot_id="B0CI", subtype="bot_message", username="ci-bot")
    fake.reply(ts(1), ts(8), BOT_USER, "My own answer in the thread.")
    fake.reply(ts(1), ts(9), "U2", "Also on the channel", subtype="thread_broadcast")

    rows = {row["id"]: row for row in _walk(fake, {})}

    assert set(rows) == {f"{CHANNEL}:{ts(1)}", _day_id(ts(3))}
    assert "My own answer" not in rows[f"{CHANNEL}:{ts(1)}"]["content"]
    assert "Also on the channel" in rows[f"{CHANNEL}:{ts(1)}"]["content"]
    # The day keeps the deploy note and the other bot's alert, and nothing of the join or
    # the app's own messages.
    assert rows[_day_id(ts(3))]["content"] == (
        "[2026-11-05 10:03 UTC] Ben Ortiz: Deploy is done.\nAttached: deploy.log, graph\n\n"
        "[2026-11-05 10:07 UTC] ci-bot: CI passed"
    )


def test_an_unknown_user_is_rendered_by_id_and_looked_up_once():
    fake = _workspace()
    fake.post(ts(4), "U9GONE", "first")
    fake.post(ts(5), "U9GONE", "second")

    rows = {row["id"]: row for row in _walk(fake, {})}

    assert rows[_day_id(ts(4))]["content"].endswith(
        "U9GONE: first\n\n[2026-11-05 10:05 UTC] U9GONE: second"
    )
    assert fake.methods().count("users.info") == 3  # U1, U2, U9GONE: one lookup each


def test_rendering_does_not_depend_on_the_order_slack_lists_messages_in():
    def rendered(seed: int | None) -> list[dict]:
        fake = _workspace()
        for minute in range(10, 20):
            fake.reply(ts(1), ts(minute), "U1" if minute % 2 else "U2", f"reply {minute}")
        if seed is not None:
            original = fake._conversations_replies

            def shuffled(**params):
                page = original(**params)
                random.Random(seed).shuffle(page["messages"])
                return page

            fake._conversations_replies = shuffled
        return _walk(fake, {})

    assert rendered(None) == rendered(1) == rendered(2)


# --- the incremental model ---


def test_an_unchanged_channel_yields_nothing_and_reads_no_thread():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.calls.clear()
    stats = _stats()

    assert _walk(fake, state, stats) == []
    assert fake.methods() == ["auth.test", "conversations.history"]
    assert stats["skipped"] == stats["scanned"] == 2


def test_a_new_reply_re_renders_only_its_thread():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.reply(ts(1), ts(10), "U1", "Thanks!")
    fake.calls.clear()

    rows = _walk(fake, state)

    assert _ids(rows) == [f"{CHANNEL}:{ts(1)}"]
    assert rows[0]["content"].endswith("Ana Lee: Thanks!")
    assert [params["ts"] for m, params in fake.calls if m == "conversations.replies"] == [ts(1)]


def test_an_edited_root_is_rendered_again():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.edit(ts(3), "Deploy is rolled back.", ts(11))

    rows = _walk(fake, state)

    assert _ids(rows) == [_day_id(ts(3))]
    assert "rolled back" in rows[0]["content"]


def test_a_deleted_root_is_tombstoned_after_a_complete_walk():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.delete(ts(3))  # the only message outside threads of its day
    stats = _stats()

    rows = _walk(fake, state, stats)

    assert rows == [{"id": _day_id(ts(3)), "_deleted": True}]
    assert stats["deleted"] == 1 and stats["failed"] == 0
    assert set(state["roots"]) == {ts(1)}


def test_a_cut_walk_forgets_nothing_and_the_walk_that_completes_it_does():
    fake = _six_messages()
    state: dict = {}
    _walk(fake, state)
    fake.delete(_at(2, 0))  # on the last page, above its oldest message

    cut = _stats()
    assert _walk(fake, state, cut, max_requests=2) == []  # auth.test and one page
    assert cut["failed_budget"] == 1 and cut["deleted"] == 0
    assert _at(2, 0) in state["roots"], "a walk cut short must not forget anything"

    resumed = _stats()
    rows = _walk(fake, state, resumed)
    assert rows == [{"id": _day_id(_at(2, 0)), "_deleted": True}]
    assert resumed["failed"] == 0


def test_a_root_found_gone_before_a_cut_is_forgotten_when_the_walk_completes():
    """A thread: a day is rendered again as soon as the walk has read it whole, but a
    thread found gone waits for the walk to complete."""
    fake = _six_messages()
    fake.reply(_at(6, 0), _at(6, 1), "U1", "a reply")
    state: dict = {}
    _walk(fake, state)
    fake.delete(_at(6, 1))
    fake.delete(_at(6, 0))  # the newest root, found gone by the first page

    cut = _stats()
    assert _walk(fake, state, cut, max_requests=3) == []  # auth.test and two pages
    assert cut["failed_budget"] == 1 and _at(6, 0) in state["roots"]

    rows = _walk(fake, state)
    assert rows == [{"id": f"{CHANNEL}:{_at(6, 0)}", "_deleted": True}]
    assert set(state) == {"v", "roots", "days"}


def test_an_unchanged_walk_leaves_the_state_exactly_as_it_found_it():
    """dlt writes a new state version, about 1 MB for 100k roots, whenever the state
    changes, so an idle sync must change nothing."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", "ana")
    for minute in range(1, 7):
        fake.post(ts(minute), "U1", f"message {minute}")
    state: dict = {}
    _walk(fake, state)
    before = json.dumps(state, sort_keys=True)

    assert _walk(fake, state) == []
    assert json.dumps(state, sort_keys=True) == before


def test_a_budget_cut_resumes_from_the_end_of_the_day_it_reached():
    fake = _six_messages()
    state: dict = {}

    first = _stats()
    # auth.test, one page, users.info and conversations.info for the first row
    rows = _walk(fake, state, first, max_requests=4)
    # The page reached day 5 but not its start, so only day 6 is read whole.
    assert _ids(rows) == [_day_id(_at(6, 0))]
    assert first["failed"] == 1 and first["failed_budget"] == 1
    assert set(first) == STATS_KEYS | {"failed_budget"}
    end_of_day_5 = f"{MIDNIGHT + 6 * DAY}.000000"
    assert state["position"] == end_of_day_5

    fake.calls.clear()
    rest = _walk(fake, state)
    assert _ids(rest) == [_day_id(_at(m, 0)) for m in (5, 4, 3, 2, 1)]
    histories = [params for method, params in fake.calls if method == "conversations.history"]
    newest_known = int(_at(6, 0).split(".")[0])
    assert histories[0]["oldest"] == f"{newest_known - 1}.999999", (
        "first what is newer than the newest known root (G47), not its whole day again (G71)"
    )
    assert histories[1]["latest"] == end_of_day_5, (
        "the resumed walk reads the day the cut reached from its end"
    )
    assert "position" not in state, "a complete walk starts the next one at the newest message"


def test_a_rate_limit_stops_the_run_cleanly_and_keeps_its_progress():
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", "ana")
    for day in range(1, 5):
        fake.post(_at(day, 0), "U1", f"message {day}")
    fake.fail("conversations.history", rate_limited(), after=1)
    state: dict = {}
    stats = _stats()

    rows = _walk(fake, state, stats)

    assert _ids(rows) == [_day_id(_at(4, 0))]
    assert stats["failed"] == 1 and stats["failed_rate_limit"] == 1
    assert set(stats) == STATS_KEYS | {"failed_rate_limit"}
    fake.heal()
    assert _ids(_walk(fake, state)) == [_day_id(_at(m, 0)) for m in (3, 2, 1)]


def test_a_thread_slack_reports_gone_is_tombstoned():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.reply(ts(1), ts(12), "U1", "late reply")
    fake.fail("conversations.replies", SlackAPIError("conversations.replies", "thread_not_found"))

    rows = _walk(fake, state)

    assert rows == [{"id": f"{CHANNEL}:{ts(1)}", "_deleted": True}]


def test_a_root_deleted_while_its_replies_stay_is_rendered_again_without_it():
    """Slack keeps the thread under a deleted-message placeholder: the reply count and the
    latest reply stay, only the subtype changes."""
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.channels[CHANNEL]["messages"][ts(1)].update(
        subtype="tombstone", text="This message was deleted."
    )

    rows = _walk(fake, state)

    assert _ids(rows) == [f"{CHANNEL}:{ts(1)}"]
    assert "launch checklist" not in rows[0]["content"]
    assert rows[0]["content"].startswith("[2026-11-05 10:02 UTC] Ben Ortiz:")


def test_a_thread_whose_messages_were_all_deleted_is_tombstoned():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.channels[CHANNEL]["messages"][ts(1)].update(subtype="tombstone", text="deleted")
    fake.delete(ts(2))
    fake.reply(ts(1), ts(13), BOT_USER, "only the app is left")

    rows = _walk(fake, state)

    assert rows == [{"id": f"{CHANNEL}:{ts(1)}", "_deleted": True}]
    assert state["roots"][ts(1)].startswith("~"), "kept as empty, so it is not read again"


def test_a_deleted_reply_that_is_not_the_latest_re_renders_its_thread():
    """Only reply_count changes: latest_reply and the root's edit stay."""
    fake = _workspace()
    fake.reply(ts(1), ts(14), "U1", "Found it, thanks.")
    state: dict = {}
    _walk(fake, state)
    fake.delete(ts(2))

    rows = _walk(fake, state)

    assert _ids(rows) == [f"{CHANNEL}:{ts(1)}"]
    assert "drive folder" not in rows[0]["content"]


def test_a_thread_that_renders_nothing_is_not_read_again_until_it_changes():
    fake = _workspace()
    fake.post(ts(20), BOT_USER, "Cognee's own answer")
    fake.reply(ts(20), ts(21), BOT_USER, "and its follow-up")
    state: dict = {}
    assert f"{CHANNEL}:{ts(20)}" not in _ids(_walk(fake, state))
    fake.calls.clear()

    assert _walk(fake, state) == []
    assert "conversations.replies" not in fake.methods()

    fake.reply(ts(20), ts(22), "U1", "a person answers the app")
    rows = _walk(fake, state)
    assert _ids(rows) == [f"{CHANNEL}:{ts(20)}"]
    assert rows[0]["content"].endswith("Ana Lee: a person answers the app")


def test_an_empty_thread_that_disappears_is_dropped_without_a_tombstone():
    """It never had a row, so the complete walk that misses it forgets it silently."""
    fake = _workspace()
    fake.post(ts(20), BOT_USER, "Cognee's own answer")
    fake.reply(ts(20), ts(21), BOT_USER, "and its follow-up")
    state: dict = {}
    _walk(fake, state)
    assert state["roots"][ts(20)].startswith("~")
    fake.delete(ts(20))
    fake.delete(ts(21))
    stats = _stats()

    assert _walk(fake, state, stats) == []
    assert stats["deleted"] == 0
    assert ts(20) not in state["roots"]


def test_a_join_without_replies_is_kept_so_its_day_is_not_rendered_on_every_sync():
    """A message without replies that the state does not hold marks its day to render; were
    the join left out of the state, every sync would render its day again."""
    fake = _workspace()
    fake.post(ts(4), "U1", "ana has joined the channel", subtype="channel_join")
    state: dict = {}
    _walk(fake, state)

    assert ts(4) in state["roots"]
    assert _walk(fake, state) == []


def test_slack_unavailable_after_rows_stops_cleanly_and_before_rows_raises():
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", "ana")
    for day in range(1, 5):
        fake.post(_at(day, 0), "U1", f"message {day}")
    outage = SlackUnavailableError("conversations.history", "http_503")
    fake.fail("conversations.history", outage, after=1)
    state: dict = {}
    stats = _stats()

    rows = _walk(fake, state, stats)

    assert _ids(rows) == [_day_id(_at(4, 0))]
    assert stats["failed"] == 1 and stats["failed_unavailable"] == 1
    assert state["position"] == f"{MIDNIGHT + 4 * DAY}.000000", (
        "the next run reads the day it reached, day 3, from its end"
    )

    fake.fail("conversations.history", outage)
    with pytest.raises(SlackUnavailableError):
        _walk(fake, {})


@pytest.mark.parametrize("code", ["internal_error", "http_400", "fatal_error"])
def test_any_other_slack_error_after_rows_keeps_what_the_run_reached(code):
    """F-B4: an error Slack names with another code raised after rows, and dlt dropped the
    state the run reached, so the next run read it all again. After rows it now cuts the run
    like an outage; before any row it still raises."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", "ana")
    for day in range(1, 5):
        fake.post(_at(day, 0), "U1", f"message {day}")
    error = SlackAPIError("conversations.history", code)
    fake.fail("conversations.history", error, after=1)
    state: dict = {}
    stats = _stats()

    rows = _walk(fake, state, stats)

    assert _ids(rows) == [_day_id(_at(4, 0))]
    assert stats["failed"] == 1 and stats["failed_unavailable"] == 1
    assert state["position"] == f"{MIDNIGHT + 4 * DAY}.000000"

    fake.fail("conversations.history", error)
    with pytest.raises(SlackAPIError):
        _walk(fake, {})


def test_a_missing_scope_after_rows_still_raises():
    """A missing scope stops the whole workspace with the reconnect, not a resumable cut."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", "ana")
    for day in range(1, 5):
        fake.post(_at(day, 0), "U1", f"message {day}")
    fake.fail(
        "conversations.history",
        SlackMissingScopeError("conversations.history", "channels:history"),
        after=1,
    )

    with pytest.raises(SlackMissingScopeError):
        _walk(fake, {})


def test_a_walk_resumed_under_new_messages_always_moves_and_completes():
    """G71: while a walk is unfinished each run first reads what is newer. That read
    rendered its days only once it had read up to the present, and a cut started it again
    from the present: once more was newer than one run could read, the channel emitted
    nothing again and the walk below never moved (measured, under a run of 8 requests and
    3 messages a page). G73: such a cut is no day too large for a run."""
    fake = FakeSlack(page_limit=3)
    fake.add_user("U1", "ana")
    for day in range(10):
        for minute in range(3):
            fake.post(_at(day, minute), "U1", f"day {day} message {minute}")
    state: dict = {}
    emitted = {row["id"] for row in _walk(fake, state, max_requests=6)}
    assert state.get("position"), "the first run is cut short"
    for day in range(10, 20):
        for minute in range(3):
            fake.post(_at(day, minute), "U1", f"day {day} message {minute}")

    for _ in range(30):
        stats = _stats()
        emitted |= {row["id"] for row in _walk(fake, state, stats, max_requests=8)}
        assert not stats.get("failed_day_too_large"), "no single day is bigger than a run"
        if "position" not in state:
            break

    assert "position" not in state, "the walk completes"
    assert emitted >= {_day_id(_at(day, 0)) for day in range(20)}


def test_a_busy_newest_day_does_not_starve_the_walk_below():
    """G71, reopened 2026-11-11: while a walk is unfinished each run first reads what is
    newer, from the start of the newest known root's day. With nothing new, that day is
    read whole again on every run; when it is busy, it spends nearly the whole run, the walk
    below never finishes an older day and the run is reported as a day too large for a run,
    though every day fits in one. Five quiet days of 4 messages, a newest day of 12, two
    messages a page, 10 requests a run (one day at most needs 6): the walk must complete and
    every day reach memory."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for day in range(5):
        for hour in range(1, 5):
            fake.post(_at(day, hour * 60), "U1", f"Old day {day}, message {hour}.")
    for minute in range(1, 13):
        fake.post(_at(5, minute), "U1", f"Busy day, message {minute}.")
    state: dict = {}
    memory: dict[str, dict] = {}

    stalled = 0
    for _ in range(30):
        stats = _stats()
        for row in _walk(fake, state, stats, max_requests=10):
            if row["_deleted"]:
                memory.pop(row["id"], None)
            else:
                memory[row["id"]] = row
        stalled += stats.get("failed_day_too_large", 0)
        if "position" not in state:
            break

    assert "position" not in state, "the walk completes"
    assert stalled == 0, "no day here is bigger than a run"
    assert set(memory) == {_day_id(_at(day, 60)) for day in range(6)}


def test_a_run_that_reads_new_threads_is_never_reported_as_stuck():
    """G83: a day of threads is walked over several runs, each reading two or three of them,
    while ``position`` stays at that day's end. Those runs emit rows, so none is a day no
    run can read whole: the flag (and its six-hour wait) is only for a run that made no
    progress at all."""
    fake = FakeSlack(page_limit=4)
    fake.add_user("U1", real_name="Ana Lee")
    for day in range(3):
        for thread in range(1, 7):
            root = _at(day, thread * 10)
            fake.post(root, "U1", f"Question {day}.{thread}?")
            fake.reply(root, _at(day, thread * 10 + 1), "U1", "Answer.")
    state: dict = {}

    for _ in range(20):
        stats = _stats()
        rows = _walk(fake, state, stats, max_requests=6)
        if rows:
            assert not stats.get("failed_day_too_large"), "a run that emitted rows made progress"
        if "position" not in state and not stats["failed"]:
            break
    assert "position" not in state and not stats["failed"], "the walk completes"


def test_a_message_that_gets_a_reply_during_a_run_leaves_its_day_and_can_be_deleted():
    """G89: while a walk is unfinished each run reads what is newer, first from the newest
    known root, then, when that finds a new message in that root's day, the whole day. A
    message the first read saw without replies, answered before the second read, was kept in
    its day's document as well as in its new thread; deleting it in Slack then removed the
    thread, and the day's document kept its text for good."""

    class AnsweredMidRun(FakeSlack):
        reply_before_the_day_read: tuple[str, str, str] | None = None

        def call(self, method, **params):
            pending = self.reply_before_the_day_read
            if (
                pending
                and method == "conversations.history"
                and params.get("oldest", "").endswith(".999999")
                and int(params["oldest"].split(".")[0]) + 1 == _day_start_of(pending[0])
            ):
                self.reply(*pending)
                self.reply_before_the_day_read = None
            return super().call(method, **params)

    def _day_start_of(ts: str) -> int:
        seconds = int(ts.split(".")[0])
        return seconds - (seconds - MIDNIGHT) % DAY

    fake = AnsweredMidRun(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for day in range(10):
        for hour in range(1, 5):
            fake.post(_at(day, hour * 60), "U1", f"Old day {day}, message {hour}.")
    fake.post(_at(10, 1), "U1", "First message today.")
    state: dict = {}
    memory: dict[str, str] = {}

    def run(max_requests: int) -> dict:
        stats = _stats()
        for row in _walk(fake, state, stats, max_requests=max_requests):
            if row["_deleted"]:
                memory.pop(row["id"], None)
            else:
                memory[row["id"]] = row["content"]
        return stats

    run(6)
    assert "position" in state, "the walk is unfinished"
    secret = _at(10, 4)
    fake.post(secret, "U1", "Secret plan.")
    fake.reply_before_the_day_read = (secret, _at(10, 5), "U1", "Reply to the plan.")
    run(8)
    for _ in range(60):
        stats = run(40)
        if "position" not in state and not stats["failed"]:
            break

    assert "Secret plan." in memory[f"{CHANNEL}:{secret}"], "it is its own thread now"
    assert "Secret plan." not in memory.get(_day_id(secret), ""), "and no longer in its day"

    fake.delete(secret)
    for _ in range(60):
        stats = run(40)
        if "position" not in state and not stats["failed"]:
            break
    assert not [row_id for row_id, text in memory.items() if "Secret plan." in text]


def test_a_workspace_address_outside_slack_com_is_refused():
    fake = _workspace()
    fake.url = "https://acme.slack.com.evil.example/"

    with pytest.raises(SlackSourceError, match="slack.com"):
        _walk(fake, {})


def test_a_root_older_than_what_slack_still_shows_is_forgotten():
    """A person deleting the oldest messages, a retention policy and a free plan's hidden
    history look the same to the API; memory follows what Slack shows (G1)."""
    fake = _six_messages()
    state: dict = {}
    _walk(fake, state)
    fake.delete(_at(1, 0))
    fake.delete(_at(2, 0))  # past the history Slack shows now
    stats = _stats()

    assert _walk(fake, state, stats) == [
        {"id": _day_id(_at(2, 0)), "_deleted": True},
        {"id": _day_id(_at(1, 0)), "_deleted": True},
    ]
    assert stats["deleted"] == 2
    assert set(state["roots"]) == {_at(day, 0) for day in range(3, 7)}


def test_a_walk_cut_before_the_last_page_forgets_nothing_past_it():
    fake = _six_messages()
    state: dict = {}
    _walk(fake, state)
    fake.delete(_at(1, 0))
    fake.delete(_at(2, 0))

    cut = _stats()
    assert _walk(fake, state, cut, max_requests=2) == []  # auth.test and the first page
    assert cut["failed_budget"] == 1 and {_at(1, 0), _at(2, 0)} <= set(state["roots"])

    assert _walk(fake, state) == [
        {"id": _day_id(_at(2, 0)), "_deleted": True},
        {"id": _day_id(_at(1, 0)), "_deleted": True},
    ], "the walk that completes forgets them"


def test_a_root_slack_shows_again_is_read_again():
    """Hidden history that comes back (a free workspace upgraded) is ingested again."""
    fake = _six_messages()
    state: dict = {}
    _walk(fake, state)
    hidden = fake.channels[CHANNEL]["messages"].pop(_at(1, 0))
    _walk(fake, state)
    fake.channels[CHANNEL]["messages"][_at(1, 0)] = hidden

    assert _ids(_walk(fake, state)) == [_day_id(_at(1, 0))]


def test_a_root_missing_inside_what_slack_shows_is_tombstoned():
    fake = _six_messages()
    state: dict = {}
    _walk(fake, state)
    fake.delete(_at(2, 0))  # _at(1, 0) is still shown, so _at(2, 0) was deleted

    assert _walk(fake, state) == [{"id": _day_id(_at(2, 0)), "_deleted": True}]


def test_a_complete_walk_that_shows_nothing_forgets_everything():
    """An empty channel is not an uncertain answer: Slack shows nothing, so memory keeps
    nothing. Were Slack wrong, the next walk reads the messages again."""
    fake = _six_messages()
    state: dict = {}
    _walk(fake, state)
    for day in range(1, 7):
        fake.delete(_at(day, 0))
    stats = _stats()

    assert _ids(_walk(fake, state, stats)) == [_day_id(_at(m, 0)) for m in range(6, 0, -1)]
    assert stats["deleted"] == 6 and state["roots"] == {}


def test_the_wall_clock_budget_cuts_the_run_and_the_next_run_resumes():
    fake = _six_messages()
    now = [0.0]
    original = fake.call

    def slow_call(method, **params):
        now[0] += 100.0  # every request takes 100 s of wall clock
        return original(method, **params)

    fake.call = slow_call
    state: dict = {}
    stats = _stats()

    rows = _walk(fake, state, stats, max_seconds=450, clock=lambda: now[0])

    # auth.test, page 1, users.info, conversations.info, page 2: the next would start at 500 s.
    # Page 2 reached day 3 but not its start, so days 6 to 4 are the ones read whole.
    assert _ids(rows) == [_day_id(_at(m, 0)) for m in (6, 5, 4)]
    assert stats["failed"] == 1 and stats["failed_budget"] == 1
    assert stats["requests"] == 5
    rest = _walk(fake, state, max_seconds=450, clock=lambda: now[0])
    assert _ids(rest) == [_day_id(_at(m, 0)) for m in (3, 2, 1)]


def test_a_source_run_has_the_wall_clock_budget_by_default():
    assert slack_module.MAX_RUN_SECONDS == 600
    assert slack_module._iter_rows.__kwdefaults__["max_seconds"] == slack_module.MAX_RUN_SECONDS


def test_history_pagination_that_does_not_advance_raises():
    fake = _workspace()
    fake._conversations_history = lambda **_: {"messages": [], "has_more": True}

    with pytest.raises(SlackAPIError, match="pagination_stalled"):
        _walk(fake, {})


def test_an_unreadable_history_page_raises_instead_of_looking_empty():
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake._conversations_history = lambda **_: {"has_more": False}

    with pytest.raises(SlackAPIError, match="invalid_response"):
        _walk(fake, state)
    assert len(state["roots"]) == 2


# --- failures ---


def test_a_missing_users_read_scope_fails_fast_naming_the_reconnect():
    fake = _workspace()
    fake.fail("users.info", missing_scope())

    with pytest.raises(SlackMissingScopeError, match="Reconnect Slack to grant users:read"):
        _walk(fake, {})


def test_an_expired_token_after_rows_stops_cleanly_and_before_rows_raises():
    fake = _workspace()
    fake.post(ts(4), "U2", "Retro at four?")
    fake.reply(ts(4), ts(5), "U1", "Yes.")
    expired = SlackAuthError("conversations.replies", "token_expired")
    fake.fail("conversations.replies", expired, after=1)
    stats = _stats()
    rows = _walk(fake, {}, stats)
    # The newest thread is read and emitted; the token expires on the next thread, ts(1).
    assert _ids(rows) == [f"{CHANNEL}:{ts(4)}"]
    assert stats["failed_auth"] == 1

    fake.fail("auth.test", SlackAuthError("auth.test", "invalid_auth"))
    with pytest.raises(SlackAuthError):
        _walk(fake, {})


def test_a_token_of_another_workspace_is_refused_before_anything_is_read():
    fake = _workspace()

    with pytest.raises(SlackSourceError, match="another workspace"):
        list(_iter_rows(fake, "T0OTHER", CHANNEL, {}, _stats()))
    assert fake.methods() == ["auth.test"]


def test_check_active_runs_before_every_request():
    fake = _workspace()
    seen = []
    _walk(fake, {}, check_active=lambda: seen.append(len(fake.calls)))
    assert seen == list(range(len(fake.calls)))


def test_the_factory_refuses_bad_arguments_without_a_request():
    fake = _workspace()
    good = {"service": fake, "team_id": TEAM, "channel_id": CHANNEL, "resource_name": TABLE}
    for bad in ({"channel_id": "general"}, {"channel_id": "D012"}, {"team_id": " "}):
        with pytest.raises(ValueError):
            slack_source(**{**good, **bad})
    with pytest.raises(ValueError):
        slack_source(team_id=TEAM, channel_id=CHANNEL, resource_name=TABLE)
    with pytest.raises(ValueError):
        slack_source(**{**good, "resource_name": ""})
    assert fake.calls == []


def test_an_older_core_is_refused(monkeypatch):
    monkeypatch.setattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 2)
    with pytest.raises(RuntimeError, match="node sets and structure"):
        slack_source(service=_workspace(), team_id=TEAM, channel_id=CHANNEL, resource_name=TABLE)


def test_a_state_of_an_unknown_version_is_refused():
    with pytest.raises(SlackSourceError, match="unknown version"):
        _walk(_workspace(), {"v": 2, "roots": {}})


def test_the_resource_carries_the_document_contract():
    source = slack_source(
        service=_workspace(), team_id=TEAM, channel_id=CHANNEL, resource_name=TABLE
    )
    assert getattr(source, dlt_utils.DOCUMENT_SOURCE_ATTR) == "slack"
    assert getattr(source, dlt_utils.PIPELINE_SCOPE_ATTR) == TABLE
    assert source.name == TABLE


# --- through a real dlt pipeline ---


def _pipeline(tmp_path):
    import dlt

    return dlt.pipeline(
        pipeline_name="slack_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 's.db'}"),
        dataset_name="slack",
        pipelines_dir=str(tmp_path / "pipelines"),
    )


def _source(fake, name: str = TABLE, channel: str = CHANNEL, **kwargs):
    return slack_source(
        service=fake, team_id=TEAM, channel_id=channel, resource_name=name, **kwargs
    )


def _run(tmp_path, source) -> dict[str, str]:
    """Run ``source`` the way ingest_dlt_source does; return the staged ``id -> content``."""
    pipeline = _pipeline(tmp_path)
    pipeline.run(source, columns=DOCUMENT_COLUMN_HINTS)
    query = f'SELECT id, content FROM "{source.name}" ORDER BY id'
    with pipeline.sql_client() as sql, sql.execute_query(query) as cursor:
        return {row[0]: row[1] for row in cursor.fetchall()}


def test_first_sync_resync_new_reply_and_delete_through_a_real_pipeline(tmp_path):
    fake = _workspace()
    first = _source(fake)
    assert list(_run(tmp_path, first)) == [f"{CHANNEL}:{ts(1)}", _day_id(ts(3))]
    assert first.cognee_sync_stats["emitted"] == 2 and first.cognee_sync_stats["failed"] == 0

    quiet = _source(fake)
    _run(tmp_path, quiet)
    assert quiet.cognee_sync_stats["emitted"] == 0
    assert quiet.cognee_sync_stats["skipped"] == quiet.cognee_sync_stats["scanned"] == 2

    # ts(3) gets a reply: it leaves its day for a thread of its own, and the day, left
    # with no message, goes.
    fake.reply(ts(3), ts(10), "U1", "Thanks!")
    reply = _source(fake)
    staged = _run(tmp_path, reply)
    assert list(staged) == [f"{CHANNEL}:{ts(1)}", f"{CHANNEL}:{ts(3)}"]
    assert reply.cognee_sync_stats["emitted"] == 1 and reply.cognee_sync_stats["deleted"] == 1
    assert staged[f"{CHANNEL}:{ts(3)}"].endswith("Ana Lee: Thanks!")

    fake.delete(ts(3))
    fake.delete(ts(10))
    forget = _source(fake)
    assert list(_run(tmp_path, forget)) == [f"{CHANNEL}:{ts(1)}"]
    assert forget.cognee_sync_stats["deleted"] == 1


def test_an_unchanged_resync_writes_no_new_dlt_state(tmp_path):
    fake = _workspace()
    _run(tmp_path, _source(fake))
    _run(tmp_path, _source(fake))

    query = 'SELECT version FROM "_dlt_pipeline_state"'
    with _pipeline(tmp_path).sql_client() as sql, sql.execute_query(query) as cursor:
        versions = cursor.fetchall()
    assert len(versions) == 1


def test_node_set_and_structure_reach_the_destination(tmp_path):
    _run(tmp_path, _source(_workspace()))
    with _pipeline(tmp_path).sql_client() as sql:
        query = f'SELECT cognee_node_set, cognee_structure FROM "{TABLE}"'
        with sql.execute_query(query) as cursor:
            stored = cursor.fetchall()

    assert {json.dumps(json.loads(row[0])) for row in stored} == {
        json.dumps([f"slack:channel:{CHANNEL}"])
    }
    assert json.loads(stored[0][1])["ancestors"][0]["name"] == "#general"


def test_a_cut_walk_keeps_its_state_across_real_runs_and_then_tombstones(tmp_path):
    fake = _six_messages()
    _run(tmp_path, _source(fake))
    fake.delete(_at(2, 0))

    cut = _source(fake, max_requests=2)
    assert len(_run(tmp_path, cut)) == 6, "a cut walk keeps every row"
    assert cut.cognee_sync_stats["failed_budget"] == 1

    rest = _source(fake)
    assert len(_run(tmp_path, rest)) == 5
    assert rest.cognee_sync_stats["deleted"] == 1


def test_a_raised_missing_scope_rolls_the_state_back(tmp_path):
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = _workspace()
    fake.fail("users.info", missing_scope())
    with pytest.raises(PipelineStepFailed) as raised:
        _run(tmp_path, _source(fake))
    assert "Reconnect Slack to grant users:read" in str(raised.value)

    fake.heal()
    retried = _source(fake)
    assert len(_run(tmp_path, retried)) == 2
    assert retried.cognee_sync_stats["emitted"] == 2


def test_two_channels_in_one_dataset_keep_their_own_tables_and_state(tmp_path):
    fake = _workspace()
    fake.add_channel("C0RANDOM", "random")
    fake.post(ts(4), "U1", "lunch?", channel="C0RANDOM")

    _run(tmp_path, _source(fake))
    other = _run(tmp_path, _source(fake, "slack_channel_other", "C0RANDOM"))
    assert list(other) == [_day_id(ts(4), "C0RANDOM")]

    fake.delete(ts(4), channel="C0RANDOM")
    _run(tmp_path, _source(fake, "slack_channel_other", "C0RANDOM"))
    assert len(_run(tmp_path, _source(fake))) == 2, "the other channel's sync deletes nothing here"


def test_the_module_imports_no_client_library_at_import_time():
    assert "httpx" not in vars(slack_module)


@pytest.mark.parametrize(
    "error",
    [
        SlackAuthError("auth.test", "token_revoked"),
        SlackUnavailableError("auth.test", "http_503"),
        SlackAPIError("auth.test", "internal_error"),
    ],
    ids=["auth", "unavailable", "api_error"],
)
def test_a_run_that_raises_before_any_row_raises_its_error_uncounted(error):
    """A raised run is reported by its error; the host counts it once, under its key
    (``test_a_channel_that_fails_before_any_row_is_counted_once``), so the stats do not."""
    fake = _workspace()
    fake.fail("auth.test", error)
    stats = _stats()

    with pytest.raises(type(error)):
        _walk(fake, {}, stats)

    assert stats["failed"] == 0


def test_a_walk_over_several_runs_asks_slack_for_each_author_once():
    """Under Tier 1 a first sync takes hundreds of runs; asking users.info again in every
    run spent most of the request budget (G50). Names live in the state while a walk is
    unfinished and go when it completes, so the next walk reads them fresh."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for minute in range(1, 5):
        fake.post(ts(minute), "U1", f"Note {minute}.")
    state: dict = {}

    _walk(fake, state, max_requests=4)  # auth.test, a page, users.info, info: cut
    assert "position" in state
    fake.calls.clear()
    _walk(fake, state)

    assert [method for method, _ in fake.calls].count("users.info") == 0
    assert "position" not in state and "names" not in state, "the walk completed"


def test_a_thread_marked_changed_is_read_again_though_its_root_looks_the_same():
    """A reply edited, or a file removed, leaves the root's fingerprint as it was (G6); the
    event handler marks the thread, and the walk reads it again and drops the mark."""
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.edit(ts(2), "In the shared drive folder.", edited_ts=ts(9))
    assert _walk(fake, state) == [], "the root looks the same"

    changed = {ts(1), ts(77)}  # ts(77) is not a root of this channel
    rows = _walk(fake, state, changed_roots=changed)

    assert [row["id"] for row in rows] == [f"{CHANNEL}:{ts(1)}"]
    assert rows[0]["content"].endswith("In the shared drive folder.")
    assert changed == set(), "read again, and a mark for no root is dropped by a complete walk"


def test_a_message_to_the_app_is_not_memory():
    """Goran, 2026-11-10: a message that starts by mentioning the app talks to the app, a
    command (`@cognee forget`, `remember`, `help`) or a question (`@cognee what did we
    decide?`); it is not what the channel said, so no document carries it. A message that
    only mentions the app later is still content."""
    from cognee.tests.unit.tasks.connectors.fake_slack import BOT_USER

    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    for minute, text in enumerate(
        [
            f"<@{BOT_USER}> forget",
            f"<@{BOT_USER}> remember buy milk",
            f"<@{BOT_USER}>  HELP",
            f"<@{BOT_USER}>",
            f"<@{BOT_USER}> link",
            f"<@{BOT_USER}> what did we decide?",
            f"thanks <@{BOT_USER}>, forget it",
        ],
        start=1,
    ):
        fake.post(_at(0, minute), "U1", text)

    (row,) = _walk(fake, {})

    assert row["content"].split("] ", 1)[1].startswith("Ana Lee: thanks"), row["content"]
    assert "buy milk" not in row["content"] and "HELP" not in row["content"]
    assert row["content"].count("] Ana Lee:") == 1, row["content"]
    assert "what did we decide?" not in row["content"] and "thanks" in row["content"]


def test_an_excluded_message_is_left_out_of_its_thread_and_its_day():
    """A message taken out of the workspace's memory is left out of every document it would
    be in, and its thread or day is rendered again without it (the handler marks it)."""
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)

    rows = {
        row["id"]: row
        for row in _walk(
            fake, state, changed_roots={ts(1), ts(3)}, excluded_messages={ts(2), ts(3)}
        )
    }

    assert "drive folder" not in rows[f"{CHANNEL}:{ts(1)}"]["content"]
    assert rows[_day_id(ts(3))] == {"id": _day_id(ts(3)), "_deleted": True}, "nothing left that day"
    # and it stays out: a later change re-renders without it
    fake.edit(ts(1), "Where is the launch list?", edited_ts=ts(8))
    (again,) = _walk(fake, state, excluded_messages={ts(2), ts(3)})
    assert "drive folder" not in again["content"]


def test_a_marked_thread_the_run_could_not_read_keeps_its_mark():
    """A run cut while reading the marked thread has not read it: the mark stays, so the
    next sync reads it (G57)."""
    fake = _workspace()
    state: dict = {}
    _walk(fake, state)
    fake.edit(ts(2), "In the shared drive folder.", edited_ts=ts(9))
    fake.fail("conversations.replies", rate_limited("conversations.replies"))

    changed = {ts(1)}
    assert _walk(fake, state, changed_roots=changed) == []
    assert changed == {ts(1)}

    fake.failures.clear()
    rows = _walk(fake, state, changed_roots=changed)
    assert rows[0]["content"].endswith("In the shared drive folder.")
    assert changed == set()


def test_a_channel_shared_with_another_organization_is_synced_and_marked():
    """Slack Connect: synced like any channel, and its documents carry ``slack:shared`` so a
    search can leave a partner's messages out (Goran, 2026-11-09)."""
    fake = _workspace()
    fake.channels[CHANNEL]["is_ext_shared"] = True

    rows = _walk(fake, {})

    assert rows and all(
        row[dlt_utils.NODE_SET_COLUMN] == [f"slack:channel:{CHANNEL}", "slack:shared"]
        for row in rows
    )


def test_an_ordinary_channel_carries_only_its_own_node_set():
    rows = _walk(_workspace(), {})
    assert all(row[dlt_utils.NODE_SET_COLUMN] == [f"slack:channel:{CHANNEL}"] for row in rows)


def test_a_message_posted_while_a_walk_is_unfinished_is_read_by_the_next_run():
    """Each run first reads what is newer than the newest known root, and the whole day of a
    message found in that root's day, then resumes the walk where the last one stopped: a
    new message does not wait for the walk to end (G47)."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for day in range(1, 7):
        fake.post(_at(day, 0), "U1", f"Note {day}.")
    state: dict = {}
    first = _walk(fake, state, max_requests=4)  # auth.test, a page, users.info, info: cut
    assert "position" in state and first

    fake.post(_at(6, 30), "U1", "Posted while the walk was unfinished.")
    # auth.test, what is newer than the newest known root (finds the new message in its
    # day), that day read whole, info: cut. A quiet channel spends one request on it (G71).
    second = _walk(fake, state, max_requests=4)

    # The newest day is read whole again, so its document keeps the earlier note too.
    assert [row["id"] for row in second] == [_day_id(_at(6, 0))]
    assert second[0]["content"] == (
        "[2026-11-11 10:00 UTC] Ana Lee: Note 6.\n\n"
        "[2026-11-11 10:30 UTC] Ana Lee: Posted while the walk was unfinished."
    )
    assert "position" in state, "the older part of the walk is still to come"


# --- G7: the messages outside threads of one day are one document ---


def test_a_days_messages_outside_threads_are_one_document_and_a_thread_its_own():
    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(_at(0, 1), "U1", "When is the release?")
    fake.post(_at(0, 2), "U1", "Friday, after the freeze.")
    fake.post(_at(0, 3), "U1", "Who owns the checklist?")
    fake.reply(_at(0, 3), _at(0, 4), "U1", "Ben does.")
    fake.post(_at(1, 1), "U1", "Release done.")

    rows = {row["id"]: row for row in _walk(fake, {})}

    assert set(rows) == {_day_id(_at(0, 1)), f"{CHANNEL}:{_at(0, 3)}", _day_id(_at(1, 1))}
    day = rows[_day_id(_at(0, 1))]
    assert day["content"] == (
        "[2026-11-05 10:01 UTC] Ana Lee: When is the release?\n\n"
        "[2026-11-05 10:02 UTC] Ana Lee: Friday, after the freeze."
    )
    assert day["title"] == "When is the release? — Ana Lee, 2026-11-05"


def test_a_message_deleted_from_a_day_renders_the_day_again_and_an_empty_day_goes():
    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(_at(0, 1), "U1", "Keep this.")
    fake.post(_at(0, 2), "U1", "Delete this.")
    fake.post(_at(1, 1), "U1", "Alone on its day.")
    state: dict = {}
    _walk(fake, state)

    fake.delete(_at(0, 2))
    fake.delete(_at(1, 1))
    rows = {row["id"]: row for row in _walk(fake, state)}

    assert rows[_day_id(_at(0, 1))]["content"] == "[2026-11-05 10:01 UTC] Ana Lee: Keep this."
    assert rows[_day_id(_at(1, 1))] == {"id": _day_id(_at(1, 1)), "_deleted": True}
    assert _walk(fake, state) == [], "nothing changed since"


def test_a_walk_cut_in_the_middle_of_a_day_renders_that_day_whole_later():
    """A cut keeps its position at the end of the day it reached, so the next run reads that
    day from its first message and no message of it is left out."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for minute in range(1, 6):
        fake.post(_at(0, minute), "U1", f"Note {minute}.")
    state: dict = {}

    first = _walk(fake, state, max_requests=2)  # auth.test and one page: cut mid-day
    assert first == [] and "position" in state
    rows = _walk(fake, state)

    assert [row["id"] for row in rows] == [_day_id(_at(0, 1))]
    assert rows[0]["content"].count("Note ") == 5
    assert "dirty_days" not in state and "position" not in state


def test_a_day_no_run_can_read_whole_is_reported_instead_of_retried_silently():
    """G56, option 1 (Goran, 2026-11-10): a cut keeps its position at the end of a day, so
    a day bigger than one run (Tier 1: about 140 messages) is restarted on every run. The
    walk cannot move, and the run says so instead of looking like any other cut."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for minute in range(1, 8):
        fake.post(_at(0, minute), "U1", f"Note {minute}.")
    state: dict = {}

    first, again = _stats(), _stats()
    _walk(fake, state, first, max_requests=3)  # auth.test and two pages: cut mid-day
    position = state["position"]
    _walk(fake, state, again, max_requests=3)

    assert "failed_day_too_large" not in first, "the first cut is an ordinary cut"
    assert state["position"] == position
    assert again["failed_day_too_large"] == 1 and again["failed_budget"] == 1


def test_a_cut_walk_that_moves_is_never_reported_as_stuck():
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for day in range(1, 7):
        fake.post(_at(day, 0), "U1", f"Day {day}.")
    state: dict = {}

    runs = []
    while len(runs) < 20:
        stats = _stats()
        # auth.test, conversations.info and two pages: every run reads one more day
        _walk(fake, state, stats, max_requests=4)
        runs.append(stats)
        if "position" not in state:
            break

    assert len(runs) > 2
    assert not [stats for stats in runs if "failed_day_too_large" in stats]


def test_a_message_that_gets_replies_leaves_its_day_and_becomes_a_thread():
    fake = FakeSlack()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(_at(0, 1), "U1", "First.")
    fake.post(_at(0, 2), "U1", "Second.")
    state: dict = {}
    _walk(fake, state)

    fake.reply(_at(0, 2), _at(0, 3), "U1", "A reply.")
    rows = {row["id"]: row for row in _walk(fake, state)}

    assert rows[_day_id(_at(0, 1))]["content"] == "[2026-11-05 10:01 UTC] Ana Lee: First."
    assert rows[f"{CHANNEL}:{_at(0, 2)}"]["content"].endswith("Ana Lee: A reply.")


# -- Review 2026-11-11 (F2, F5): red until fixed --------------------------------------


class _BadThread(FakeSlack):
    """Slack answers one thread's replies with a page that is not a list, every time."""

    bad_root: str | None = None

    def _conversations_replies(self, channel: str, ts: str, limit: int, cursor: str = ""):
        if ts == self.bad_root:
            return {"messages": "not a list"}
        return super()._conversations_replies(channel, ts, limit, cursor)


def _two_threads_the_older_one_bad() -> _BadThread:
    fake = _BadThread()
    fake.add_user("U1", real_name="Ana Lee")
    fake.post(_at(-1, 0), "U1", "A note from the day before.")
    fake.post(ts(1), "U1", "Old question?")
    fake.reply(ts(1), ts(2), "U1", "Old answer.")
    fake.post(ts(5), "U1", "New question?")
    fake.reply(ts(5), ts(6), "U1", "New answer.")
    fake.bad_root = ts(1)
    return fake


def test_a_malformed_answer_after_rows_keeps_the_rows_and_counts_the_failure():
    """F2: a malformed Slack answer (here a replies page that is not a list) was raised
    outside ``_call``, so after rows dlt dropped every row the run emitted and the stats
    said ``failed=0``. It must follow the rule ``_call`` applies to any other Slack error
    after rows (F-B4): the rows stay and the failure is counted."""
    fake = _two_threads_the_older_one_bad()
    stats = _stats()

    rows = _walk(fake, {}, stats)

    assert f"{CHANNEL}:{ts(5)}" in _ids(rows)
    assert stats["failed"] >= 1


def test_a_thread_slack_always_answers_badly_does_not_stop_the_walk_for_good():
    """F2: the same malformed answer on every run stuck the channel at that thread: nothing
    older was ever read, and nothing was ever forgotten. The walk must get past it (the
    thread counted as failed) and read what is older."""
    fake = _two_threads_the_older_one_bad()
    state: dict = {}
    emitted: list[str] = []
    for _ in range(3):
        try:
            emitted += _ids(_walk(fake, state))
        except SlackAPIError:
            pass

    assert _day_id(_at(-1, 0)) in emitted


def test_the_stall_warning_names_no_channel(caplog):
    """F5: the warning for a day no run can read whole logged the channel id; record ids
    must not reach logs (the module docstring and ``slack.md`` Privacy)."""
    fake = FakeSlack(page_limit=2)
    fake.add_user("U1", real_name="Ana Lee")
    for minute in range(1, 8):
        fake.post(_at(0, minute), "U1", f"Note {minute}.")
    state: dict = {}
    _walk(fake, state, _stats(), max_requests=3)

    with caplog.at_level("WARNING"):
        stats = _stats()
        _walk(fake, state, stats, max_requests=3)

    assert stats["failed_day_too_large"] == 1
    assert CHANNEL not in caplog.text
