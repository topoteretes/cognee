# Adapted from topoteretes/cognee-community (Apache-2.0), PR #370, commit 5e78950.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Sync state machine tests: _iter_rows against an in-memory Google Calendar.

The contract: a failing calendar never forgets anything and never blocks the
others, an event is forgotten only if it reached memory, one real calendar is
one set of documents however it is named, and no log line or error text carries
a calendar id. No network, no LLM. Each test drives the fake through a
real-world change (edit, delete, cancelled occurrence, expired sync token,
broken calendar, ...) and checks the rows the connector yields and the state it
keeps for the next run.
"""

import logging
from datetime import datetime, timezone

import pytest
from fake_calendar import FakeCalendar, HttpError, install_google_errors

from cognee.tasks.ingestion.connectors.google_calendar import (
    GoogleCalendarError,
    _iter_rows,
    _plain_text,
)

START = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)
PRIMARY = "lay@gmail.com"
TEAM = "team@group.calendar.google.com"
# Looks like what it is in real accounts: an email address nobody wants in a log.
SECRET_TEAM = "secret-roadmap@group.calendar.google.com"


@pytest.fixture(autouse=True)
def _google_errors(monkeypatch):
    install_google_errors(monkeypatch)


@pytest.fixture
def cal():
    return FakeCalendar()


def sync(fake, state, calendars=("primary",), check_active=None):
    stats: dict = {}
    rows = list(_iter_rows(fake, tuple(calendars), state, stats, check_active))
    upserts = {row["id"]: row for row in rows if not row["_deleted"]}
    deletes = sorted(row["id"] for row in rows if row["_deleted"])
    return upserts, deletes, stats


def rid(event_id, calendar=PRIMARY):
    return f"gcal:{calendar}:{event_id}"


def snapshot(state):
    return {"calendars": {k: dict(v) for k, v in state["calendars"].items()}}


# ---------------------------------------------------------------------------
# First sync, paging and request shape
# ---------------------------------------------------------------------------
def test_first_sync_reads_every_page_and_saves_the_token(cal):
    ids = {cal.put_event("primary", f"Meeting {i}", START) for i in range(5)}
    state: dict = {}

    upserts, deletes, stats = sync(cal, state)

    assert set(upserts) == {rid(i) for i in ids}
    assert deletes == []
    assert stats == {"scanned": 5, "skipped": 0, "failed": 0, "deleted": 0, "calendars": 1}
    assert state["calendars"][PRIMARY]["sync_token"] == f"{PRIMARY}~5"
    assert sorted(state["calendars"][PRIMARY]["known"]) == sorted(ids)


def test_sync_token_rounds_keep_the_same_parameters_on_every_page(cal):
    cal.put_event("primary", "One", START)
    state: dict = {}
    sync(cal, state)
    for i in range(5):
        cal.put_event("primary", f"New {i}", START)
    cal.list_calls.clear()

    sync(cal, state)

    first, *pages = cal.list_calls
    assert first["syncToken"] == f"{PRIMARY}~1"
    assert pages, "the change round spans several pages"
    for page in pages:
        assert {k: v for k, v in page.items() if k != "pageToken"} == first
    assert "timeMin" not in first and "timeMax" not in first


def test_event_row_shape_and_content(cal):
    event_id = cal.put_event(
        "primary",
        "Battery review",
        START,
        location="Pune office",
        attendees=[
            {
                "email": "arjun@example.com",
                "displayName": "Arjun Rao",
                "responseStatus": "accepted",
            },
            {"email": "lay@gmail.com", "responseStatus": "tentative"},
        ],
        hangoutLink="https://meet.google.com/abc-defg-hij",
        description="<p>Battery lasts <b>55 minutes</b>.</p>",
    )

    upserts, _, _ = sync(cal, {})

    assert upserts[rid(event_id)] == {
        "id": rid(event_id),
        "title": "Battery review",
        "content": (
            "Calendar event: Battery review\n"
            "Calendar: Lay Patel\n"
            "When: 2026-10-12T10:00:00+05:30 (Asia/Kolkata) to "
            "2026-10-12T11:00:00+05:30 (Asia/Kolkata)\n"
            "Location: Pune office\n"
            "Organizer: Meera Joshi <meera@example.com>\n"
            "Attendees: Arjun Rao <arjun@example.com> (accepted), lay@gmail.com (tentative)\n"
            "Meeting link: https://meet.google.com/abc-defg-hij\n\n"
            "Battery lasts 55 minutes."
        ),
        "url": f"https://calendar.google.com/event?eid={event_id}",
        "_deleted": False,
    }


def test_attendee_order_from_google_does_not_change_the_content(cal):
    # The content hash decides whether the event is re-cognified, so a reshuffle
    # by Google must not look like an edit.
    ada = {"email": "ada@example.com", "responseStatus": "accepted"}
    bob = {"email": "bob@example.com", "responseStatus": "declined"}
    nameless = {"displayName": "Zed", "responseStatus": "needsAction"}
    first = cal.put_event("primary", "Review", START, attendees=[ada, bob, nameless])
    second = cal.put_event("primary", "Review", START, attendees=[nameless, bob, ada])

    upserts, _, _ = sync(cal, {})

    assert upserts[rid(first)]["content"] == upserts[rid(second)]["content"]
    content = upserts[rid(first)]["content"]
    assert "Zed (needsAction), ada@example.com (accepted), bob@example.com (declined)" in content


def test_omitted_attendees_are_said_so(cal):
    big = cal.put_event(
        "primary",
        "All hands",
        START,
        attendees=[{"email": "a@example.com", "responseStatus": "accepted"}],
        attendeesOmitted=True,
    )

    upserts, _, _ = sync(cal, {})

    assert "and more not listed by Google" in upserts[rid(big)]["content"]


def test_all_day_event_ends_on_its_last_day_not_the_exclusive_end_date(cal):
    # Google sends end.date as the day after the event, so a three-day event
    # starting on the 8th arrives with end.date 2026-11-11.
    trip = cal.put_all_day("primary", "Offsite", "2026-11-08", days=3)

    upserts, _, _ = sync(cal, {})

    content = upserts[rid(trip)]["content"]
    assert "When: all day, 2026-11-08 to 2026-11-10\n" in content
    assert "2026-11-11" not in content


def test_single_day_all_day_event_has_no_end(cal):
    holiday = cal.put_all_day("primary", "Diwali", "2026-11-08")

    upserts, _, _ = sync(cal, {})

    assert "When: all day, 2026-11-08\n" in upserts[rid(holiday)]["content"]
    assert " to " not in upserts[rid(holiday)]["content"].split("\n")[2]


def test_working_location_events_are_skipped(cal):
    meeting = cal.put_event("primary", "Sync", START)
    cal.put_event("primary", "Office", START, eventType="workingLocation")

    upserts, _, stats = sync(cal, {})

    assert set(upserts) == {rid(meeting)}
    assert stats["skipped"] == 1


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("description", "expected"),
    [
        # Real tags go, the text stays.
        ("<p>Battery lasts <b>55 minutes</b>.</p>", "Battery lasts 55 minutes."),
        ('<a href="https://example.com/doc">the doc</a><br/>Next', "the doc\nNext"),
        ("<ul><li>one</li><li>two</li></ul>", "one\ntwo"),
        ("Fish &amp; chips", "Fish & chips"),
        # Angle brackets that are the user's own text stay.
        ("Budget <5 lakh> approved", "Budget <5 lakh> approved"),
        ("Keep a < b > c", "Keep a < b > c"),
        ("Mail <arjun@example.com> the notes", "Mail <arjun@example.com> the notes"),
        # An escaped tag is text, not markup.
        ("Use &lt;b&gt; for bold", "Use <b> for bold"),
    ],
)
def test_plain_text_strips_html_tags_and_only_those(description, expected):
    assert _plain_text(description) == expected


# ---------------------------------------------------------------------------
# Incremental rounds, edits and deletes
# ---------------------------------------------------------------------------
def test_resync_without_changes_yields_nothing(cal):
    cal.put_event("primary", "Sync", START)
    state: dict = {}
    sync(cal, state)

    upserts, deletes, _ = sync(cal, state)

    assert upserts == {} and deletes == []


def test_edit_reyields_the_same_event(cal):
    event_id = cal.put_event("primary", "Draft agenda", START)
    state: dict = {}
    sync(cal, state)

    cal.edit_event("primary", event_id, summary="Final agenda")
    upserts, deletes, _ = sync(cal, state)

    assert list(upserts) == [rid(event_id)]
    assert upserts[rid(event_id)]["title"] == "Final agenda"
    assert deletes == []


def test_deleted_event_is_forgotten(cal):
    keep = cal.put_event("primary", "Keep", START)
    gone = cal.put_event("primary", "Budget meeting", START)
    state: dict = {}
    sync(cal, state)

    cal.delete_event("primary", gone)
    upserts, deletes, stats = sync(cal, state)

    assert upserts == {}
    assert deletes == [rid(gone)]
    assert stats["deleted"] == 1
    assert state["calendars"][PRIMARY]["known"] == [keep]


def test_event_created_and_deleted_between_two_syncs_is_not_tombstoned(cal):
    # It never reached memory, so there is nothing to forget and nothing to count.
    keep = cal.put_event("primary", "Keep", START)
    state: dict = {}
    sync(cal, state)

    blip = cal.put_event("primary", "Blip", START)
    cal.delete_event("primary", blip)
    upserts, deletes, stats = sync(cal, state)

    assert upserts == {}
    assert deletes == []
    assert stats["deleted"] == 0
    assert state["calendars"][PRIMARY]["known"] == [keep]


# ---------------------------------------------------------------------------
# Recurring events
# ---------------------------------------------------------------------------
def test_series_is_one_document_with_its_rule(cal):
    series = cal.put_series("primary", "Falcon daily sync", START, "RRULE:FREQ=DAILY;COUNT=3")

    upserts, _, _ = sync(cal, {})

    assert list(upserts) == [rid(series)]
    assert "Repeats: RRULE:FREQ=DAILY;COUNT=3" in upserts[rid(series)]["content"]


def test_cancelled_single_occurrence_never_forgets_the_series(cal):
    series = cal.put_series("primary", "Falcon daily sync", START, "RRULE:FREQ=DAILY;COUNT=3")
    state: dict = {}
    sync(cal, state)

    cal.cancel_occurrence("primary", series, START.replace(day=13))
    _, deletes, _ = sync(cal, state)

    assert deletes == []
    assert state["calendars"][PRIMARY]["known"] == [series]


def test_changed_occurrence_is_its_own_document_and_can_be_cancelled(cal):
    series = cal.put_series("primary", "Falcon daily sync", START, "RRULE:FREQ=DAILY;COUNT=3")
    state: dict = {}
    sync(cal, state)

    moved = cal.edit_occurrence("primary", series, START.replace(day=13), location="Rooftop")
    upserts, _, _ = sync(cal, state)
    assert list(upserts) == [rid(moved)]
    assert f"Changed occurrence of series {series}" in upserts[rid(moved)]["content"]

    cal.cancel_occurrence("primary", series, START.replace(day=13))
    upserts, deletes, _ = sync(cal, state)
    assert deletes == [rid(moved)]
    assert state["calendars"][PRIMARY]["known"] == [series]


def test_deleting_a_whole_series_forgets_its_changed_occurrences(cal):
    series = cal.put_series("primary", "Falcon daily sync", START, "RRULE:FREQ=DAILY;COUNT=3")
    moved = cal.edit_occurrence("primary", series, START.replace(day=13), location="Rooftop")
    state: dict = {}
    sync(cal, state)

    cal.delete_event("primary", series)
    _, deletes, _ = sync(cal, state)

    assert deletes == sorted([rid(series), rid(moved)])


# ---------------------------------------------------------------------------
# Expired tokens and calendars that leave
# ---------------------------------------------------------------------------
def test_expired_sync_token_resyncs_like_a_snapshot(cal):
    keep = cal.put_event("primary", "Keep", START)
    gone = cal.put_event("primary", "Gone", START)
    state: dict = {}
    sync(cal, state)

    cal.delete_event("primary", gone)
    cal.expired_tokens.add(state["calendars"][PRIMARY]["sync_token"])
    upserts, deletes, _ = sync(cal, state)

    assert deletes == [rid(gone)]
    assert set(upserts) == {rid(keep)}  # a full listing re-yields every event


def test_calendar_that_left_the_calendar_list_forgets_its_events(cal):
    cal.add_calendar(TEAM, "Team")
    team = cal.put_event(TEAM, "Team lunch", START)
    state: dict = {}
    calendars = ("primary", TEAM)
    sync(cal, state, calendars)

    cal.delete_calendar(TEAM)
    _upserts, deletes, _ = sync(cal, state, calendars)

    assert deletes == [rid(team, TEAM)]
    assert set(state["calendars"]) == {PRIMARY}


def test_dropping_a_calendar_from_the_config_forgets_its_events(cal):
    cal.add_calendar(TEAM, "Team")
    team = cal.put_event(TEAM, "Team lunch", START)
    state: dict = {}
    sync(cal, state, ("primary", TEAM))

    _, deletes, _ = sync(cal, state, ("primary",))

    assert deletes == [rid(team, TEAM)]
    assert set(state["calendars"]) == {PRIMARY}


def test_404_for_a_calendar_still_in_the_calendar_list_forgets_nothing(cal):
    # The same run saw the calendar in calendarList, so the 404 is not "gone".
    cal.add_calendar(TEAM, "Team")
    cal.put_event(TEAM, "Team lunch", START)
    state: dict = {}
    sync(cal, state, ("primary", TEAM))
    before = snapshot(state)
    cal.expired_tokens.add(state["calendars"][TEAM]["sync_token"])
    cal.event_errors[TEAM] = 404

    upserts, deletes, stats = sync(cal, state, ("primary", TEAM))

    assert upserts == {} and deletes == []
    assert stats["deleted"] == 0 and stats["failed"] == 1
    assert state == before


def test_404_for_the_only_calendar_still_in_the_list_raises(cal):
    cal.put_event("primary", "Keep", START)
    state: dict = {}
    sync(cal, state)
    before = snapshot(state)
    cal.event_errors[PRIMARY] = 404

    with pytest.raises(GoogleCalendarError, match="HTTP 404"):
        sync(cal, state)

    assert state == before


def test_404_for_a_calendar_that_was_never_synced_is_a_failure_not_a_forget(cal):
    # Most likely a typo in the config.
    keep = cal.put_event("primary", "Keep", START)
    state: dict = {}

    upserts, deletes, stats = sync(cal, state, ("primary", "typo@group.calendar.google.com"))

    assert set(upserts) == {rid(keep)}
    assert deletes == []
    assert stats["failed"] == 1 and stats["failed_calendar_sync"] == 1
    assert set(state["calendars"]) == {PRIMARY}


def test_a_config_with_only_a_never_synced_calendar_raises(cal):
    with pytest.raises(GoogleCalendarError, match="HTTP 404"):
        sync(cal, {}, ("typo@group.calendar.google.com",))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_errors_raise_and_never_delete(cal, status):
    cal.put_event("primary", "Keep", START)
    state: dict = {}
    sync(cal, state)
    before = snapshot(state)

    cal.event_errors[PRIMARY] = status
    with pytest.raises(GoogleCalendarError, match=f"HTTP {status}"):
        sync(cal, state)

    assert state == before


def test_error_on_a_later_page_saves_nothing(cal):
    cal.put_event("primary", "One", START)
    state: dict = {}
    sync(cal, state)
    for i in range(5):
        cal.put_event("primary", f"New {i}", START)
    before = snapshot(state)

    original = cal._list_events

    def fail_on_second_page(params):
        if "pageToken" in params:
            raise HttpError(503)
        return original(params)

    cal._list_events = fail_on_second_page
    with pytest.raises(GoogleCalendarError, match="HTTP 503"):
        sync(cal, state)

    assert state == before


def test_a_round_without_next_sync_token_fails_the_calendar(cal):
    cal.put_event("primary", "Keep", START)
    original = cal._list_events

    def no_token(params):
        response = original(params)
        response.pop("nextSyncToken", None)
        return response

    cal._list_events = no_token
    with pytest.raises(GoogleCalendarError, match="nextSyncToken"):
        sync(cal, {})


def test_calendar_failing_after_another_finished_keeps_the_finished_one(cal):
    cal.add_calendar(TEAM, "Team")
    keep = cal.put_event("primary", "Keep", START)
    cal.put_event(TEAM, "Team lunch", START)
    cal.event_errors[TEAM] = 500
    state: dict = {}

    upserts, deletes, stats = sync(cal, state, ("primary", TEAM))

    assert set(upserts) == {rid(keep)}
    assert deletes == []
    assert set(state["calendars"]) == {PRIMARY}
    assert state["calendars"][PRIMARY]["sync_token"] == f"{PRIMARY}~1"
    assert stats["calendars"] == 1 and stats["failed"] == 1
    assert stats["failed_calendar_sync"] == 1


def test_failed_calendar_keeps_its_old_state_and_is_retried(cal):
    cal.add_calendar(TEAM, "Team")
    team = cal.put_event(TEAM, "Team lunch", START)
    state: dict = {}
    sync(cal, state, ("primary", TEAM))
    old_team_state = dict(state["calendars"][TEAM])

    cal.event_errors[TEAM] = 503
    cal.put_event("primary", "New", START)
    _, deletes, _ = sync(cal, state, ("primary", TEAM))
    assert deletes == []
    assert state["calendars"][TEAM] == old_team_state

    del cal.event_errors[TEAM]
    cal.delete_event(TEAM, team)
    _, deletes, _ = sync(cal, state, ("primary", TEAM))
    assert deletes == [rid(team, TEAM)]


def test_every_calendar_failing_raises_and_saves_nothing(cal):
    cal.add_calendar(TEAM, "Team")
    cal.event_errors[PRIMARY] = 500
    cal.event_errors[TEAM] = 500
    state: dict = {}

    with pytest.raises(GoogleCalendarError, match=r"all 2 configured calendar\(s\).*HTTP 500"):
        sync(cal, state, ("primary", TEAM))

    assert state == {}


@pytest.mark.parametrize(
    ("status", "key"),
    [(401, "failed_auth"), (429, "failed_rate_limit"), (500, "failed_calendar_sync")],
)
def test_a_dying_run_says_why_in_the_stats(cal, status, key):
    stats: dict = {}
    cal.event_errors[PRIMARY] = status

    with pytest.raises(GoogleCalendarError):
        list(_iter_rows(cal, ("primary",), {}, stats))

    assert stats["failed"] == 1 and stats[key] == 1
    assert {k for k in stats if k.startswith("failed")} == {"failed", key}


def test_a_failing_calendar_list_is_counted_too(cal):
    stats: dict = {}
    cal.fail_next.append(429)

    with pytest.raises(GoogleCalendarError, match="HTTP 429"):
        list(_iter_rows(cal, ("primary",), {}, stats))

    assert stats["failed"] == 1 and stats["failed_rate_limit"] == 1


@pytest.mark.parametrize(
    ("status", "reason"), [(401, None), (429, None), (403, "rateLimitExceeded")]
)
def test_auth_and_rate_limit_end_the_run_but_keep_what_finished(cal, status, reason):
    # Every later calendar would fail the same way, so none is tried.
    if reason:
        cal.event_error_reasons["a@group.calendar.google.com"] = reason
    cal.add_calendar("a@group.calendar.google.com", "A")
    cal.add_calendar("b@group.calendar.google.com", "B")
    keep = cal.put_event("primary", "Keep", START)
    cal.event_errors["a@group.calendar.google.com"] = status
    state: dict = {}

    upserts, _, stats = sync(
        cal, state, ("primary", "a@group.calendar.google.com", "b@group.calendar.google.com")
    )

    assert set(upserts) == {rid(keep)}
    assert set(state["calendars"]) == {PRIMARY}
    assert stats["failed"] == 1
    assert [c["calendarId"] for c in cal.list_calls] == [PRIMARY, "a@group.calendar.google.com"]


@pytest.mark.parametrize("reason", ["rateLimitExceeded", "userRateLimitExceeded"])
def test_a_403_with_a_rate_limit_reason_is_counted_as_rate_limit(cal, reason):
    stats: dict = {}
    cal.event_errors[PRIMARY] = 403
    cal.event_error_reasons[PRIMARY] = reason

    with pytest.raises(GoogleCalendarError, match="HTTP 403"):
        list(_iter_rows(cal, ("primary",), {}, stats))

    assert stats["failed_rate_limit"] == 1 and "failed_calendar_sync" not in stats


def test_a_403_without_a_rate_limit_reason_fails_only_its_calendar(cal):
    # Lost access to one calendar says nothing about the others.
    cal.add_calendar(TEAM, "Team")
    cal.add_calendar("b@group.calendar.google.com", "B")
    keep = cal.put_event("b@group.calendar.google.com", "Keep", START)
    cal.event_errors[TEAM] = 403
    cal.event_error_reasons[TEAM] = "forbidden"

    upserts, _, stats = sync(cal, {}, (TEAM, "b@group.calendar.google.com"))

    assert set(upserts) == {rid(keep, "b@group.calendar.google.com")}
    assert stats["failed"] == 1 and stats["failed_calendar_sync"] == 1
    assert "failed_rate_limit" not in stats


def test_stats_start_every_run_with_zero_failures(cal):
    cal.put_event("primary", "One", START)
    stats: dict = {"failed_auth": 1, "scanned": 9}

    list(_iter_rows(cal, ("primary",), {}, stats))

    assert stats == {"scanned": 1, "skipped": 0, "failed": 0, "deleted": 0, "calendars": 1}


# ---------------------------------------------------------------------------
# Authorization checkpoint
# ---------------------------------------------------------------------------
class _Revoked(Exception):
    pass


def test_check_active_runs_before_every_api_call_not_only_between_rows(cal):
    for i in range(5):
        cal.put_event("primary", f"Meeting {i}", START)  # 3 pages of 2
    state: dict = {}

    def check_active():
        # The connection is revoked after calendarList and the first page.
        if len(cal.api_calls) >= 2:
            raise _Revoked

    with pytest.raises(_Revoked):
        sync(cal, state, check_active=check_active)

    assert cal.api_calls == ["calendarList", "events"], "no request after the revocation"
    assert state == {}


def test_check_active_is_not_swallowed_by_the_per_calendar_error_handling(cal):
    cal.add_calendar(TEAM, "Team")
    cal.put_event("primary", "Keep", START)
    calls: list[int] = []

    def check_active():
        calls.append(1)
        if len(calls) == 3:  # calendarList, the primary round, then the team round
            raise _Revoked

    with pytest.raises(_Revoked):
        sync(cal, {}, ("primary", TEAM), check_active=check_active)


# ---------------------------------------------------------------------------
# Several calendars
# ---------------------------------------------------------------------------
def test_events_in_two_calendars_keep_separate_ids_and_names(cal):
    cal.add_calendar(TEAM, "Team")
    shared_id = "invite1"
    cal.put_event("primary", "Launch", START, id=shared_id)
    cal.put_event(TEAM, "Launch", START, id=shared_id)

    upserts, _, _ = sync(cal, {}, ("primary", TEAM))

    assert set(upserts) == {rid(shared_id), rid(shared_id, TEAM)}
    assert "Calendar: Team" in upserts[rid(shared_id, TEAM)]["content"]


def test_primary_and_its_real_id_are_one_calendar(cal):
    event = cal.put_event("primary", "Launch", START)
    state: dict = {}

    upserts, _, stats = sync(cal, state, ("primary", PRIMARY))

    assert list(upserts) == [rid(event)]
    assert set(state["calendars"]) == {PRIMARY}
    assert stats["scanned"] == 1 and stats["calendars"] == 1
    assert len(cal.list_calls) == 1, "one round for one calendar"


def test_primary_keeps_its_real_id_when_the_list_has_it(cal):
    # If the list ever lacked the primary calendar (a hidden one), "primary" would
    # stay literal and every row id would change.
    event = cal.put_event("primary", "Launch", START)
    state: dict = {}

    upserts, _, _ = sync(cal, state)

    assert list(upserts) == [rid(event)]
    assert set(state["calendars"]) == {PRIMARY}


def test_primary_and_its_real_id_later_in_the_config_are_the_same_state(cal):
    event = cal.put_event("primary", "Launch", START)
    state: dict = {}
    sync(cal, state, ("primary",))

    upserts, deletes, _ = sync(cal, state, (PRIMARY,))

    assert upserts == {} and deletes == []
    assert state["calendars"][PRIMARY]["known"] == [event]


# ---------------------------------------------------------------------------
# No calendar id in logs or errors
# ---------------------------------------------------------------------------
def test_no_calendar_id_reaches_logs_or_error_text(cal, caplog):
    caplog.set_level(logging.DEBUG)
    cal.add_calendar(SECRET_TEAM, "Roadmap")
    cal.put_event("primary", "Keep", START)
    cal.put_event(SECRET_TEAM, "Secret plan", START)
    state: dict = {}
    sync(cal, state, ("primary", SECRET_TEAM))

    # Every path that logs: an expired token relisting, a broken calendar,
    # a calendar that left the list, then a run where everything fails.
    cal.expired_tokens.add(state["calendars"][SECRET_TEAM]["sync_token"])
    sync(cal, state, ("primary", SECRET_TEAM))
    cal.event_errors[SECRET_TEAM] = 500
    sync(cal, state, ("primary", SECRET_TEAM))
    del cal.event_errors[SECRET_TEAM]
    cal.delete_calendar(SECRET_TEAM)
    sync(cal, state, ("primary", SECRET_TEAM))
    cal.event_errors[PRIMARY] = 404
    with pytest.raises(GoogleCalendarError) as error:
        sync(cal, state, ("primary", "typo-" + SECRET_TEAM))

    assert caplog.text, "the paths above log"
    for calendar_id in (PRIMARY, SECRET_TEAM):
        assert calendar_id not in caplog.text
        assert calendar_id not in str(error.value)
    # The cause is chained for debugging and carries the same promise.
    chain = error.value.__cause__
    assert chain is not None and SECRET_TEAM not in str(chain) and PRIMARY not in str(chain)
