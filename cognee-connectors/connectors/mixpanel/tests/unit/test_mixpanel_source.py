"""The sync itself: what rows a run emits, how it stays incremental, how it forgets.

Driven end to end through :func:`cognee_connector_mixpanel.source._iter_rows` with a
:class:`~fake_mixpanel.FakeMixpanel` as the service and a plain dict as the dlt resource
state, the way the dlt resource drives it inside ``remember``.
"""

from datetime import date, datetime, timezone

import pytest
from fake_mixpanel import PROJECT_ID, FakeMixpanel

from cognee_connector_mixpanel.source import (
    MixpanelRateLimitedError,
    MixpanelSourceError,
    _iter_rows,
)

TODAY = date(2026, 11, 16)
# 2026-11-10T12:00:00Z and 2026-11-11T12:00:00Z: two events on the 10th, one on the 11th
NOON_10 = int(datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc).timestamp())
NOON_11 = int(datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc).timestamp())

DEFAULT_KINDS = ("annotations", "cohorts", "funnels", "schemas")


def run(fake, state, *, include=DEFAULT_KINDS, events_from=None, project_id=None, today=TODAY):
    stats: dict[str, int] = {}
    rows = list(
        _iter_rows(
            fake,
            state,
            stats,
            include=include,
            project_id=project_id,
            events_from=events_from,
            today=lambda: today,
        )
    )
    return rows, stats


def seeded(fake: FakeMixpanel) -> FakeMixpanel:
    fake.add_annotation("a1", "Launched the new onboarding funnel")
    fake.add_cohort("c1", "Power users", description="Users with 10+ sessions a week")
    fake.add_funnel("f1", "Signup to first project", ["Signup", "Project created"])
    fake.add_schema("Signup", description="A user signed up", properties=["plan", "referrer"])
    return fake


def row_ids(rows):
    return [row["id"] for row in rows]


# --- the ingest path -------------------------------------------------------


def test_a_first_run_emits_every_definition_as_a_document_row():
    fake = seeded(FakeMixpanel())

    rows, stats = run(fake, {})

    assert row_ids(rows) == [
        "annotation:a1",
        "cohort:c1",
        "funnel:f1",
        "schema:Signup",
    ]
    assert stats["scanned"] == 4 and stats["skipped"] == 0
    by_id = {row["id"]: row for row in rows}
    assert by_id["cohort:c1"]["title"] == "Power users"
    assert "Power users" in by_id["cohort:c1"]["content"]
    assert by_id["cohort:c1"]["content"].startswith("Mixpanel cohort Power users.")
    assert by_id["cohort:c1"]["url"] == f"https://mixpanel.com/project/{PROJECT_ID}/cohorts"
    assert by_id["cohort:c1"]["_deleted"] is False
    assert by_id["cohort:c1"]["cognee_node_set"] == [f"mixpanel:{PROJECT_ID}:cohorts"]
    # the funnel renders its steps in order: the order is the definition
    assert "Steps: Signup; Project created" in by_id["funnel:f1"]["content"]
    # the schema renders sorted properties
    assert "Properties: plan, referrer" in by_id["schema:Signup"]["content"]
    # the volatile cohort count is never rendered
    assert "42" not in by_id["cohort:c1"]["content"]


def test_a_run_resolves_the_project_from_the_credential_and_pins_it_in_state():
    fake = seeded(FakeMixpanel())

    state: dict = {}
    run(fake, state)

    assert state["project_id"] == PROJECT_ID
    assert fake.count("/projects") == 1
    # a later run under an explicit project id makes no discovery request
    run(fake, state, project_id=PROJECT_ID)
    assert fake.count("/projects") == 1


def test_a_credential_that_sees_several_projects_must_be_told_which_to_sync():
    fake = FakeMixpanel()
    fake.extra_projects.append("999")

    with pytest.raises(MixpanelSourceError, match="project_id"):
        run(fake, {})


def test_a_different_project_under_the_same_resource_name_is_refused():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    with pytest.raises(MixpanelSourceError, match="another Mixpanel project"):
        run(fake, state, project_id="999")


# --- incremental: only what changed ----------------------------------------


def test_a_second_run_without_changes_emits_nothing():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    rows, stats = run(fake, state)

    assert rows == []
    assert stats["scanned"] == 4 and stats["skipped"] == 4 and stats["deleted"] == 0


def test_a_changed_definition_reemits_only_that_row():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    fake.cohorts["c1"]["description"] = "Users with 20+ sessions a week"
    rows, stats = run(fake, state)

    assert row_ids(rows) == ["cohort:c1"]
    assert "20+" in rows[0]["content"]
    assert stats["skipped"] == 3


def test_a_reordered_funnel_is_a_change_but_a_rerun_is_not():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    fake.funnels["f1"]["steps"] = [{"event": "Project created"}, {"event": "Signup"}]
    rows, _ = run(fake, state)
    assert row_ids(rows) == ["funnel:f1"]
    assert "Steps: Project created; Signup" in rows[0]["content"]

    assert run(fake, state)[0] == []


# --- forget-on-delete ------------------------------------------------------


def test_a_definition_deleted_upstream_is_forgotten_on_the_next_run():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    del fake.cohorts["c1"]
    rows, stats = run(fake, state)

    assert rows == [{"id": "cohort:c1", "_deleted": True}]
    assert stats["deleted"] == 1
    # and the tombstone does not come back on a later run
    assert run(fake, state)[0] == []


def test_a_kind_left_out_of_include_is_forgotten():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    rows, _ = run(fake, state, include=("cohorts",))

    assert sorted(row_ids(rows)) == ["annotation:a1", "funnel:f1", "schema:Signup"]
    assert all(row["_deleted"] is True for row in rows)


def test_a_failed_listing_changes_nothing_and_forgets_nothing():
    fake = seeded(FakeMixpanel())
    state: dict = {}
    run(fake, state)

    fake.fail_paths["/cohorts/list"] = MixpanelRateLimitedError(
        "Mixpanel rate limit exceeded (HTTP 429)"
    )
    with pytest.raises(MixpanelRateLimitedError):
        run(fake, state)

    # dlt would drop the state of the failed run; simulate the next run against the
    # surviving state: everything is still there and nothing is a tombstone.
    fake.fail_paths.clear()
    rows, stats = run(fake, state)
    assert rows == []
    assert stats["deleted"] == 0


def test_check_active_runs_before_every_request():
    fake = seeded(FakeMixpanel())
    calls: list[str] = []

    def check_active():
        calls.append("check")

    list(
        _iter_rows(
            fake,
            {},
            {},
            include=DEFAULT_KINDS,
            project_id=PROJECT_ID,
            today=lambda: TODAY,
            check_active=check_active,
        )
    )
    # the walker checks before each of the 4 listing requests, and the client-level
    # check runs again inside each request, the way MixpanelClient does in its retries
    assert len(calls) == 2 * 4
    assert len(fake.calls) == 4
    assert fake.calls[0] != ("GET", "/projects")  # the explicit project id skips discovery


# --- the events kind: a date-window cursor ----------------------------------


def events_fake() -> FakeMixpanel:
    fake = FakeMixpanel()
    window = ("2026-11-10", "2026-11-15")
    fake.add_event(*window, "Signup", epoch_seconds=NOON_10)
    fake.add_event(*window, "Signup", epoch_seconds=NOON_10 + 3600)
    fake.add_event(*window, "Login", epoch_seconds=NOON_10)
    fake.add_event(*window, "Signup", epoch_seconds=NOON_11)
    return fake


def test_the_first_events_run_exports_the_window_and_advances_the_cursor():
    fake = events_fake()
    state: dict = {}

    rows, stats = run(fake, state, include=("events",), events_from=date(2026, 11, 10))

    assert row_ids(rows) == [
        "events:2026-11-10:Login",
        "events:2026-11-10:Signup",
        "events:2026-11-11:Signup",
    ]
    by_id = {row["id"]: row for row in rows}
    assert "recorded 2 'Signup' events" in by_id["events:2026-11-10:Signup"]["content"]
    assert by_id["events:2026-11-11:Signup"]["cognee_node_set"] == [f"mixpanel:{PROJECT_ID}:events"]
    # the window ends the day before the run, and the cursor lands there
    assert state["events_cursor"] == "2026-11-15"
    assert stats["requests"] == 1  # one export; discovery is not a listing request


def test_the_next_events_run_exports_only_the_days_since_the_cursor():
    fake = events_fake()
    state: dict = {}
    run(fake, state, include=("events",), events_from=date(2026, 11, 10))

    # a day later, with yesterday closed, only the new day exports
    fake.add_event("2026-11-16", "2026-11-16", "Signup", epoch_seconds=NOON_11 + 5 * 86400)
    rows, _ = run(fake, state, include=("events",), today=date(2026, 11, 17))

    assert row_ids(rows) == ["events:2026-11-16:Signup"]
    assert state["events_cursor"] == "2026-11-16"


def test_a_run_on_the_same_day_exports_nothing():
    fake = events_fake()
    state: dict = {}
    run(fake, state, include=("events",), events_from=date(2026, 11, 10))

    rows, _ = run(fake, state, include=("events",), today=date(2026, 11, 16))

    assert rows == []


def test_a_first_events_run_without_events_from_fails_fast():
    fake = events_fake()

    with pytest.raises(MixpanelSourceError, match="events_from"):
        run(fake, {}, include=("events",))


def test_two_events_of_a_day_aggregate_to_one_row_and_do_not_reemit():
    fake = events_fake()
    state: dict = {}
    run(fake, state, include=("events",), events_from=date(2026, 11, 10))

    fake.add_event("2026-11-16", "2026-11-16", "Signup", epoch_seconds=NOON_11 + 5 * 86400)
    fake.add_event("2026-11-16", "2026-11-16", "Signup", epoch_seconds=NOON_11 + 5 * 86400 + 60)
    rows, _ = run(fake, state, include=("events",), today=date(2026, 11, 17))
    assert row_ids(rows) == ["events:2026-11-16:Signup"]
    assert "recorded 2 'Signup' events" in rows[0]["content"]

    assert run(fake, state, include=("events",), today=date(2026, 11, 17))[0] == []


def test_events_are_never_forgotten():
    fake = events_fake()
    state: dict = {}
    run(fake, state, include=("events",), events_from=date(2026, 11, 10))

    rows, _ = run(fake, state, include=DEFAULT_KINDS)

    assert "events:2026-11-10:Signup" not in row_ids(rows)


# --- argument validation ----------------------------------------------------


def test_include_must_name_known_kinds():
    fake = FakeMixpanel()
    with pytest.raises(ValueError, match="include"):
        run(fake, {}, include=("churn",))
    with pytest.raises(ValueError, match="include"):
        run(fake, {}, include=())
