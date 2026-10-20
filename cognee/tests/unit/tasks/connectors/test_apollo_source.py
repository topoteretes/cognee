# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A record is re-extracted only when its rendered text changes, and an error never forgets one.

Nothing the text leaves out (custom fields nobody listed, step positions, activity dates) may
move a fingerprint, or every sequence step would re-extract its contact. A run reads only the
endpoints its included kinds need. A deletion check that fails keeps the record, backs off if
it keeps failing, and says why in the counts. The walk is driven with a plain dict as dlt
resource state: no network, no LLM.
"""

import random
import re

import pytest
from fake_apollo import FakeApollo

from cognee.tasks.ingestion.connectors.apollo import (
    PAGE_SIZE,
    ApolloAccessError,
    ApolloAPIError,
    ApolloAuthError,
    ApolloRateLimitedError,
    ApolloSourceError,
    Lookups,
    _iter_rows,
    apollo_team_id,
    render_contact,
)

ROW_COLUMNS = {"id", "title", "content", "url", "_deleted"}
# What apollo.md says about a record whose deletion check keeps failing.
FAILURES_BEFORE_BACKOFF = 5
BACKOFF_CYCLES = 10
# Requests a run spends on lookups before it lists anything: contact stages, account
# stages, lists and the sequence list; a contacts-only run skips the account stages.
LOOKUPS = 4
CONTACT_LOOKUPS = 3
PHONE = "+1 555 0199"
# Every count a run reports, whichever way it ends.
COUNT_KEYS = {"scanned", "skipped", "deleted", "failed", "requests", "held_back"}
# The requests that read lookups, so a run that stops after them has listed nothing.
LOOKUP_PATHS = {"/contact_stages", "/account_stages", "/labels", "/emailer_campaigns/search"}
# Each stage or list filter, the kind it needs included and the endpoint that lists its ids.
FILTERS = [
    ("contact_stage_ids", ("contacts",), "/contact_stages"),
    ("contact_label_ids", ("contacts",), "/labels"),
    ("account_stage_ids", ("accounts",), "/account_stages"),
    ("account_label_ids", ("accounts",), "/labels"),
]


def _sync(fake: FakeApollo, state: dict, **kwargs) -> tuple[list[dict], dict]:
    stats: dict[str, int] = {}
    rows = list(_iter_rows(fake, state, stats, **kwargs))
    return rows, stats


def _ids(rows: list[dict]) -> set[str]:
    return {row["id"] for row in rows}


def _tombstones(rows: list[dict]) -> set[str]:
    return {row["id"] for row in rows if row.get("_deleted")}


def _workspace() -> FakeApollo:
    fake = FakeApollo()
    fake.add_sequence("s1", "Q4 Outreach")
    fake.add_account("a1", "Acme")
    fake.add_account("a2", "Globex")
    fake.add_contact("c1", "Ada Lovelace", account_id="a1")
    fake.add_contact("c2", "Ben Turing", account_id="a2")
    return fake


def test_first_sync_emits_every_record_with_the_frozen_row_shape():
    rows, stats = _sync(_workspace(), {})

    assert _ids(rows) == {"sequence:s1", "account:a1", "account:a2", "contact:c1", "contact:c2"}
    assert all(set(row) == ROW_COLUMNS for row in rows)
    assert all(row["_deleted"] is False and row["title"] and row["content"] for row in rows)
    urls = {row["id"]: row["url"] for row in rows}
    assert urls["contact:c1"] == "https://app.apollo.io/#/contacts/c1"
    assert urls["account:a1"] == "https://app.apollo.io/#/accounts/a1"
    assert urls["sequence:s1"] == "https://app.apollo.io/#/sequences/s1"
    assert stats["failed"] == 0


def test_enrichment_channels_and_volatile_stats_are_never_rendered():
    rows, _ = _sync(_workspace(), {})
    text = " ".join(f"{row['title']} {row['content']}" for row in rows)

    for leaked in ("example.com/in", "linkedin", "Enriched", "555 0100", "@example.com", "5000"):
        assert leaked not in text
    assert "0.4" not in text  # sequence open rate


def test_stage_and_list_ids_are_rendered_as_names():
    fake = FakeApollo()
    fake.add_contact("c1", "Ada Lovelace", title="CTO", contact_stage_id="cs2", label_ids=["l1"])
    fake.add_account("a1", "Acme", account_stage_id="as2", label_ids=["l2"])
    rows, _ = _sync(fake, {}, include=("contacts", "accounts"))
    content = {row["id"]: row["content"] for row in rows}

    assert content["contact:c1"].splitlines() == [
        "Job title: CTO",
        "Company: Acme",
        "Stage: Interested",
        "Lists: Champions",
    ]
    assert content["account:a1"].splitlines() == [
        "Domain: a1.example.com",
        "Stage: Customer",
        "Lists: Key Accounts",
    ]


def test_a_custom_field_is_rendered_only_when_its_id_is_listed():
    """Teams keep phone numbers and personal emails in custom fields, so none are rendered
    until the caller names the ones it wants."""
    fake = FakeApollo()
    fake.add_contact("c1", "Ada Lovelace", typed_custom_fields={"f1": PHONE})
    fake.add_account("a1", "Acme", typed_custom_fields={"f3": "Gold"})

    rows, _ = _sync(fake, {}, include=("contacts", "accounts"))
    text = " ".join(row["content"] for row in rows)

    assert PHONE not in text and "Gold" not in text
    assert "Messaging Angle" not in text and "Tier" not in text
    assert fake.count("/fields") == 0, "no field was asked for, so the list is not read"

    rows, _ = _sync(
        fake, {}, include=("contacts", "accounts"), custom_field_ids=frozenset({"contact.f1"})
    )
    content = {row["id"]: row["content"] for row in rows}

    assert f"Messaging Angle: {PHONE}" in content["contact:c1"].splitlines()
    assert "Tier" not in content["account:a1"], "account.f3 was not listed"

    rows, _ = _sync(
        fake, {}, include=("contacts", "accounts"), custom_field_ids=frozenset({"account.f3"})
    )
    content = {row["id"]: row["content"] for row in rows}

    assert "Tier: Gold" in content["account:a1"].splitlines()
    assert PHONE not in content["contact:c1"], "contact.f1 was not listed"


# A field Apollo does not list, and the id of a contact field under the account modality.
@pytest.mark.parametrize("unlisted", ["contact.nope", "account.f1"])
def test_a_custom_field_id_apollo_does_not_list_fails_the_run_before_any_row(unlisted):
    fake, state = _workspace(), {}

    with pytest.raises(ApolloSourceError, match=rf"custom_field_ids.*\['{unlisted}'\]"):
        _sync(fake, state, custom_field_ids=frozenset({"contact.f1", unlisted}))

    assert state["fingerprints"] == {}


def test_an_accounts_only_run_renders_the_listed_account_custom_field():
    fake = _workspace()
    fake.accounts["a1"]["typed_custom_fields"] = {"f3": "Gold"}

    rows, _ = _sync(fake, {}, include=("accounts",), custom_field_ids=frozenset({"account.f3"}))
    content = {row["id"]: row["content"] for row in rows}

    assert "Tier: Gold" in content["account:a1"].splitlines()
    assert fake.count("/fields") == 1


def test_custom_field_ids_on_a_sequences_only_run_never_read_the_fields_endpoint():
    """Sequence rows render no custom fields, so a key without fields/index must not fail them."""
    fake = _workspace()
    fake.fail_paths["/fields"] = ApolloAccessError(
        "The Apollo API key or plan cannot call /fields (HTTP 403)"
    )

    rows, stats = _sync(
        fake, {}, include=("sequences",), custom_field_ids=frozenset({"contact.f1"})
    )

    assert _ids(rows) == {"sequence:s1"}
    assert stats["failed"] == 0
    assert fake.count("/fields") == 0


def test_an_unchanged_resync_emits_nothing():
    fake, state = _workspace(), {}
    _sync(fake, state)

    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["scanned"] == 5
    assert stats["skipped"] == 5


def test_an_edit_re_emits_only_that_record():
    fake, state = _workspace(), {}
    _sync(fake, state)

    fake.contacts["c2"]["title"] = "Head of Sales"
    rows, _ = _sync(fake, state)

    assert _ids(rows) == {"contact:c2"}
    assert "Job title: Head of Sales" in rows[0]["content"]


def test_an_enrollment_re_emits_the_contact_although_updated_at_did_not_move():
    fake, state = _workspace(), {}
    _sync(fake, state)
    updated_at = fake.contacts["c1"]["updated_at"]

    fake.enroll("c1", "s1", status="paused")
    rows, _ = _sync(fake, state)

    assert fake.contacts["c1"]["updated_at"] == updated_at
    assert _ids(rows) == {"contact:c1"}
    assert rows[0]["content"].splitlines()[-1] == "Sequences: Q4 Outreach (paused)"


def test_a_sequence_step_that_moves_only_activity_does_not_re_emit_the_contact():
    """The rendered contact once carried the dated activity feed, so every step a sequence
    sent re-extracted it. Now a sequence is its name and status, and a step changes neither."""
    fake, state = _workspace(), {}
    fake.enroll("c1", "s1")
    _sync(fake, state)

    fake.advance_step("c1")
    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["skipped"] == stats["scanned"] == 5


def test_deleted_contacts_and_accounts_are_tombstoned_after_a_check():
    fake, state = _workspace(), {}
    _sync(fake, state)

    del fake.contacts["c2"]
    del fake.accounts["a2"]
    rows, stats = _sync(fake, state)

    assert rows == [{"id": "account:a2", "_deleted": True}, {"id": "contact:c2", "_deleted": True}]
    assert stats["deleted"] == 2
    assert fake.count("/contacts/c2") == 1
    assert fake.count("/accounts/a2") == 1
    assert "contact:c2" not in state["fingerprints"]


def test_a_removed_sequence_is_tombstoned():
    fake, state = _workspace(), {}
    _sync(fake, state)

    del fake.sequences["s1"]
    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == {"sequence:s1"}


def test_a_record_missing_from_search_but_still_in_scope_is_kept():
    fake, state = _workspace(), {}
    _sync(fake, state)

    fake.hidden.add("c2")
    rows, _ = _sync(fake, state)

    assert rows == []
    assert "contact:c2" in state["fingerprints"]


def test_a_contact_that_left_the_selected_stage_is_forgotten():
    fake, state = _workspace(), {}
    _sync(fake, state, include=("contacts",), filters={"contact_stage_ids": ["cs1"]})

    fake.contacts["c2"]["contact_stage_id"] = "cs2"
    rows, _ = _sync(fake, state, include=("contacts",), filters={"contact_stage_ids": ["cs1"]})

    assert _tombstones(rows) == {"contact:c2"}


def test_a_contact_that_left_the_selected_list_is_forgotten():
    fake, state = _workspace(), {}
    filters = {"contact_label_ids": ["l1"]}
    fake.contacts["c1"]["label_ids"] = ["l1"]
    fake.contacts["c2"]["label_ids"] = ["l1", "l2"]
    _sync(fake, state, include=("contacts",), filters=filters)

    fake.contacts["c2"]["label_ids"] = ["l2"]
    fake.hidden.add("c1")  # missing from search but still in the list
    rows, _ = _sync(fake, state, include=("contacts",), filters=filters)

    assert _tombstones(rows) == {"contact:c2"}


def test_deselecting_a_kind_forgets_its_records():
    fake, state = _workspace(), {}
    _sync(fake, state)

    rows, _ = _sync(fake, state, include=("contacts",))

    assert _tombstones(rows) == {"sequence:s1", "account:a1", "account:a2"}
    assert fake.count("/accounts/a1") == 0  # deselected records need no check


# --- a run reads only what its kinds need ---
# A plan without sequences, or a key scoped to fewer endpoints, must still sync the
# kinds it can reach. A lookup that no included kind renders is never requested.


@pytest.mark.parametrize(
    ("include", "paths"),
    [
        (("accounts",), {"/contact_stages", "/account_stages", "/labels", "/accounts/search"}),
        (
            ("contacts",),
            {"/contact_stages", "/labels", "/emailer_campaigns/search", "/contacts/search"},
        ),
        # contact stages are always read: they carry the team id
        (("sequences",), {"/contact_stages", "/emailer_campaigns/search"}),
    ],
)
def test_a_run_reads_only_the_endpoints_its_kinds_need(include, paths):
    fake = _workspace()

    _sync(fake, {}, include=include)

    assert {path for _, path in fake.calls} == paths


@pytest.mark.parametrize(
    ("denied", "include"),
    [
        # a plan or a scoped key without sequences
        ("/emailer_campaigns/search", ("accounts",)),
        ("/account_stages", ("contacts", "sequences")),
        ("/labels", ("sequences",)),
        # no custom_field_ids, so the fields endpoint is never needed
        ("/fields", ("contacts", "accounts", "sequences")),
        # a plan that refuses contact search still syncs the other kinds
        ("/contacts/search", ("accounts", "sequences")),
    ],
)
def test_an_endpoint_the_key_cannot_call_spares_a_run_that_does_not_need_it(denied, include):
    fake = _workspace()
    fake.fail_paths[denied] = ApolloAccessError(
        f"The Apollo API key or plan cannot call {denied} (HTTP 403)"
    )

    rows, stats = _sync(fake, {}, include=include)

    by_kind = {
        "contacts": {"contact:c1", "contact:c2"},
        "accounts": {"account:a1", "account:a2"},
        "sequences": {"sequence:s1"},
    }
    assert _ids(rows) == set().union(*(by_kind[kind] for kind in include))
    assert stats["failed"] == 0


def test_a_plan_that_refuses_search_fails_the_run_and_forgets_nothing():
    fake, state = _workspace(), {}
    _sync(fake, state)
    before = dict(state["fingerprints"])

    fake.fail_paths["/contacts/search"] = ApolloAccessError(
        "The Apollo API key or plan cannot call /contacts/search (HTTP 403)"
    )
    with pytest.raises(ApolloAccessError, match="/contacts/search"):
        _sync(fake, state)

    assert state["fingerprints"] == before
    assert state["pending_deletes"] == []


# --- a filter that names nothing must not forget everything ---
# A stage or list id Apollo does not list matches no record, so the listing comes back empty
# and every synced record looks deleted. The run refuses the filter before it lists anything.


# A blank, a leading space and an id that is not there.
@pytest.mark.parametrize("unknown", ["", " cs1", "cs9"])
@pytest.mark.parametrize(("name", "include", "endpoint"), FILTERS)
def test_a_filter_naming_an_unknown_id_fails_the_run_before_anything_is_forgotten(
    name, include, endpoint, unknown
):
    fake, state = _workspace(), {}
    _sync(fake, state)
    before = dict(state["fingerprints"])
    calls_before = len(fake.calls)

    message = rf"{name} names ids Apollo does not list: \[{re.escape(repr(unknown))}\].*{endpoint}"
    with pytest.raises(ApolloSourceError, match=message):
        _sync(fake, state, include=include, filters={name: [unknown]})

    assert state["fingerprints"] == before
    assert state["pending_deletes"] == []
    assert {path for _, path in fake.calls[calls_before:]} <= LOOKUP_PATHS, "nothing was listed"


def test_a_bad_filter_also_stops_the_deletes_a_cut_run_left_pending():
    fake, state = _workspace(), {}
    _sync(fake, state)
    del fake.contacts["c2"]
    fake.fail_paths["/contacts/c2"] = ApolloRateLimitedError("Apollo rate limit exceeded")
    _sync(fake, state)
    assert state["pending_deletes"] == ["contact:c2"]
    del fake.fail_paths["/contacts/c2"]
    checks_before = fake.count("/contacts/c2")

    with pytest.raises(ApolloSourceError, match="contact_stage_ids"):
        _sync(fake, state, include=("contacts",), filters={"contact_stage_ids": ["cs9"]})

    assert state["pending_deletes"] == ["contact:c2"]
    assert "contact:c2" in state["fingerprints"]
    assert fake.count("/contacts/c2") == checks_before, "the pending check did not run"


def test_a_filter_naming_listed_ids_still_forgets_what_left_the_selection():
    fake, state = _workspace(), {}
    filters = {"account_stage_ids": ["as1"], "account_label_ids": ["l2"]}
    fake.accounts["a1"]["label_ids"] = ["l2"]
    fake.accounts["a2"]["label_ids"] = ["l2"]
    _sync(fake, state, include=("accounts",), filters=filters)

    fake.accounts["a2"]["account_stage_id"] = "as2"
    rows, stats = _sync(fake, state, include=("accounts",), filters=filters)

    assert _tombstones(rows) == {"account:a2"}
    assert stats["failed"] == 0


# A filter is checked for the kinds the run includes, not for whatever it happened to read:
# contact stages are always loaded, and contact_stage_ids is still not checked without contacts.
@pytest.mark.parametrize(
    ("name", "include"),
    [
        ("contact_stage_ids", ("accounts", "sequences")),
        ("contact_label_ids", ("accounts",)),
        ("account_stage_ids", ("contacts",)),
        ("account_label_ids", ("sequences",)),
    ],
)
def test_a_filter_of_a_kind_that_is_not_included_is_not_checked(name, include):
    rows, stats = _sync(_workspace(), {}, include=include, filters={name: ["cs9"]})

    assert stats["failed"] == 0
    assert rows


# --- budget, rate limit and resuming ---


def test_a_rate_limited_run_stops_cleanly_and_resumes_without_false_deletes():
    fake, state = _workspace(), {}
    _sync(fake, state)
    del fake.contacts["c2"]

    # the lookups and the accounts page pass, the contacts search is refused
    fake.fail_after = len(fake.calls) + LOOKUPS + 1
    fake.fail_with = ApolloRateLimitedError("Apollo rate limit exceeded (HTTP 429)")
    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["failed_rate_limit"] == 1
    assert "contact:c2" in state["fingerprints"]

    fake.fail_after = None
    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"contact:c2"}
    assert stats["failed"] == 0


def test_a_cut_during_listing_resumes_at_the_same_page():
    fake, state = FakeApollo(), {}
    for n in range(PAGE_SIZE * 2 + 50):
        fake.add_contact(f"c{n:03d}", f"Person {n}")
    rows, stats = _sync(fake, state, include=("contacts",), max_requests=CONTACT_LOOKUPS + 2)

    assert stats["failed_budget"] == 1
    assert len(rows) == PAGE_SIZE * 2  # the lookups, then two pages
    assert state["walks"]["contacts"]["page"] == 3

    rows, stats = _sync(fake, state, include=("contacts",))

    assert len(rows) == 50
    assert stats["failed"] == 0
    assert _tombstones(rows) == set()


def test_a_run_counts_the_requests_it_made():
    fake = _workspace()

    _, stats = _sync(fake, {})

    assert stats["requests"] == len(fake.calls)


@pytest.mark.parametrize("window", ["minute", "hourly", "daily"])
def test_low_remaining_quota_stops_the_run_before_the_next_request(window):
    fake, state = _workspace(), {}
    fake.rate_limit = {"minute": 40, "hourly": 400, "daily": 500, window: 3}

    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats["failed_rate_limit"] == 1
    assert fake.calls == []


def test_an_auth_error_before_any_row_raises():
    fake = _workspace()
    fake.fail_after = 0
    fake.fail_with = ApolloAuthError("Apollo rejected the API key (HTTP 401)")

    with pytest.raises(ApolloAuthError):
        _sync(fake, {})


def test_a_key_rejected_after_rows_were_emitted_cuts_the_run_and_keeps_them():
    fake, state = _workspace(), {}

    # the lookups, the sequence row and the accounts page pass, the contacts search is rejected
    fake.fail_after = LOOKUPS + 1
    fake.fail_with = ApolloAuthError("Apollo rejected the API key (HTTP 401)")
    rows, stats = _sync(fake, state)

    assert _ids(rows) == {"sequence:s1", "account:a1", "account:a2"}
    assert stats["failed_auth"] == 1
    assert stats["failed"] == 1
    assert set(state["fingerprints"]) == _ids(rows), "the rows already emitted are kept"


def _plain():
    return _workspace(), {}


def _cut_by_budget():
    return _workspace(), {"max_requests": LOOKUPS}


def _cut_by_quota():
    fake = _workspace()
    fake.rate_limit = {"minute": 3}
    return fake, {}


def _cut_by_rejected_key():
    fake = _workspace()
    fake.fail_after = LOOKUPS + 1
    fake.fail_with = ApolloAuthError("Apollo rejected the API key (HTTP 401)")
    return fake, {}


def _capped():
    fake = _workspace()
    fake.total_override = 60_000
    return fake, {}


@pytest.mark.parametrize(
    "scenario", [_plain, _cut_by_budget, _cut_by_quota, _cut_by_rejected_key, _capped]
)
def test_every_run_reports_every_count_key_however_it_ends(scenario):
    fake, kwargs = scenario()

    _, stats = _sync(fake, {}, **kwargs)

    assert COUNT_KEYS <= stats.keys()
    assert stats["held_back"] == 0


def test_a_capped_listing_never_tombstones():
    fake, state = _workspace(), {}
    _sync(fake, state)

    del fake.contacts["c2"]
    fake.total_override = 60_000
    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == set()
    assert stats["failed_cap"] == 1


# --- a deletion check that fails keeps the record ---


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        # a 500, an unreadable answer or any other error Apollo gave for the record
        (ApolloAPIError("Apollo request failed: HTTP 500"), "failed_deletion_check"),
        # a key or plan that cannot read one record: the fix is the key, not another try
        (
            ApolloAccessError("The Apollo API key or plan cannot call /contacts/{id}"),
            "failed_access",
        ),
    ],
)
def test_a_failing_check_keeps_that_record_and_the_sync_moves_on(error, reason):
    fake, state = _workspace(), {}
    _sync(fake, state)

    fake.hidden.add("c2")
    fake.fail_paths["/contacts/c2"] = error
    del fake.contacts["c1"]
    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"contact:c1"}
    assert stats["failed"] == 1
    assert stats[reason] == 1
    other_reason = {"failed_deletion_check", "failed_access"} - {reason}
    assert not other_reason & stats.keys()
    assert "contact:c2" in state["fingerprints"]

    fake.add_contact("c3", "Cara Hopper")
    rows, stats = _sync(fake, state)

    assert _ids(rows) == {"contact:c3"}
    assert stats["failed"] == 1
    assert "contact:c2" in state["fingerprints"]


def test_a_check_answered_without_a_record_keeps_it():
    fake, state = _workspace(), {}
    filters = {"contact_stage_ids": ["cs1"]}
    _sync(fake, state, include=("contacts",), filters=filters)

    fake.hidden.update({"c1", "c2"})
    fake.answers["/contacts/c1"] = {}
    fake.answers["/contacts/c2"] = {"contact": None}
    rows, stats = _sync(fake, state, include=("contacts",), filters=filters)

    assert rows == []
    assert stats["failed"] == 2
    assert stats["failed_deletion_check"] == 2
    assert {"contact:c1", "contact:c2"} <= set(state["fingerprints"])


# --- a record that keeps failing backs off ---
# A record whose check always fails (a 422 Apollo words differently one day, a contact the
# key cannot read) used to cost a request and a failed count on every run for ever.


def _stuck_workspace() -> tuple[FakeApollo, dict]:
    """c2 is missing from search, still in Apollo's index, and its detail call keeps failing."""
    fake, state = _workspace(), {}
    _sync(fake, state)
    fake.hidden.add("c2")
    fake.fail_paths["/contacts/c2"] = ApolloAPIError("Apollo request failed: HTTP 500")
    return fake, state


def test_a_record_whose_check_keeps_failing_backs_off_then_is_tried_again():
    fake, state = _stuck_workspace()

    checks, failed, reasons, held = [], [], [], []
    for _ in range(FAILURES_BEFORE_BACKOFF + BACKOFF_CYCLES + 1):
        before = fake.count("/contacts/c2")
        _, stats = _sync(fake, state)
        checks.append(fake.count("/contacts/c2") - before)
        failed.append(stats["failed"])
        reasons.append(stats.get("failed_deletion_check", 0))
        held.append(stats["held_back"])

    tries = [1] * FAILURES_BEFORE_BACKOFF + [0] * BACKOFF_CYCLES + [1]
    assert checks == tries, "five failed checks, ten cycles left alone, then one more try"
    assert failed == tries, "a record in back-off must not keep every run's failed count up"
    assert reasons == tries
    # a run that holds a record back is not a failed run, but it says what it left alone
    assert held == [0] * FAILURES_BEFORE_BACKOFF + [1] * BACKOFF_CYCLES + [0]
    assert "contact:c2" in state["fingerprints"], "backing off never forgets the record"


def test_a_record_that_recovers_after_its_back_off_is_forgotten():
    fake, state = _stuck_workspace()
    for _ in range(FAILURES_BEFORE_BACKOFF):
        _sync(fake, state)

    # Apollo now answers the check, and the record is gone
    del fake.contacts["c2"]
    del fake.fail_paths["/contacts/c2"]
    for _ in range(BACKOFF_CYCLES):
        rows, _ = _sync(fake, state)
        assert rows == [], "a record in back-off is not checked, so it is not forgotten yet"
    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"contact:c2"}
    assert stats["failed"] == 0
    assert state["check_failures"] == {}


def test_a_record_in_back_off_is_forgotten_at_once_when_its_kind_is_deselected():
    """Deselecting a kind is how a team removes prospects on purpose. Back-off protects a
    request that deselected kinds never make, so it must not hold the record for ten more cycles."""
    fake, state = _stuck_workspace()
    for _ in range(FAILURES_BEFORE_BACKOFF):
        _sync(fake, state)
    checks_before = fake.count("/contacts/c2")

    rows, stats = _sync(fake, state, include=("accounts", "sequences"))

    assert "contact:c2" in _tombstones(rows)
    assert fake.count("/contacts/c2") == checks_before, "a deselected kind needs no check"
    assert stats["failed"] == stats["held_back"] == 0
    assert state["check_failures"] == {}


# held_back is the records in back-off when the run starts. A run that never closes the
# listing, because it is cut or capped, never reaches the check that holds them back, yet
# they are still held.
@pytest.mark.parametrize(
    ("options", "total", "cut"),
    [({"max_requests": LOOKUPS}, None, "failed_budget"), ({}, 60_000, "failed_cap")],
)
def test_a_cut_or_capped_run_still_reports_the_records_in_back_off(options, total, cut):
    fake, state = _stuck_workspace()
    for _ in range(FAILURES_BEFORE_BACKOFF):
        _sync(fake, state)
    fake.total_override = total

    _, stats = _sync(fake, state, **options)

    assert stats[cut] == 1
    assert stats["held_back"] == 1
    assert stats["failed"] == 1, "only the cut fails the run, not the record held back"


def test_a_record_listed_again_drops_its_failure_streak():
    fake, state = _stuck_workspace()
    _sync(fake, state)
    assert set(state["check_failures"]) == {"contact:c2"}

    fake.hidden.discard("c2")
    _sync(fake, state)

    assert state["check_failures"] == {}


# --- the rest of the walk ---


@pytest.mark.parametrize("path", ["/emailer_campaigns/search", "/contacts/search"])
def test_a_listing_without_pagination_fails_instead_of_forgetting(path):
    fake, state = _workspace(), {}
    for n in range(PAGE_SIZE + 50):
        fake.add_sequence(f"s{n:03d}", f"Sequence {n}")
    _sync(fake, state)

    fake.no_pagination.add(path)
    with pytest.raises(ApolloAPIError, match="pagination"):
        _sync(fake, state)

    assert state.get("pending_deletes") == []
    assert len(state["fingerprints"]) == PAGE_SIZE + 50 + 5


def test_changing_the_filters_restarts_the_listing():
    fake, state = FakeApollo(), {}
    for n in range(PAGE_SIZE * 3):
        fake.add_contact(f"c{n:03d}", f"Person {n}", contact_stage_id="cs1" if n % 2 else "cs2")
    first, _ = _sync(
        fake,
        state,
        include=("contacts",),
        filters={"contact_stage_ids": ["cs1"]},
        max_requests=CONTACT_LOOKUPS + 1,
    )
    assert state["walks"]["contacts"]["page"] == 2

    rows, stats = _sync(fake, state, include=("contacts",), filters={"contact_stage_ids": ["cs2"]})

    assert stats["failed"] == 0
    emitted = {row["id"] for row in rows if not row["_deleted"]}
    assert emitted == {f"contact:c{n:03d}" for n in range(0, PAGE_SIZE * 3, 2)}
    # the old filter's contacts leave in the same cycle, not one later
    assert {row["id"] for row in rows if row["_deleted"]} == _ids(first)


def test_check_active_runs_before_every_request_and_can_stop_the_sync():
    fake, checks = _workspace(), []

    def check_active():
        checks.append(len(fake.calls))
        if len(checks) > 3:
            raise PermissionError("access revoked")

    with pytest.raises(PermissionError):
        _sync(fake, {}, check_active=check_active)

    assert checks == [0, 1, 2, 3]
    assert len(fake.calls) == 3


# --- one team per resource_name ---


def test_state_of_another_apollo_team_is_refused():
    state = {}
    _sync(_workspace(), state)

    other = FakeApollo(team_id="team2")
    other.add_contact("c9", "Dev Hollerith")
    with pytest.raises(ApolloSourceError, match="resource_name"):
        _sync(other, state)

    assert set(state["fingerprints"]) >= {"contact:c1", "contact:c2"}


def test_a_run_that_cannot_read_a_team_id_is_refused_before_anything_is_stored():
    """Without a team id the state check would pass for any team, and a run under the wrong
    key would forget the rows of the team that owns the state."""
    fake, state = FakeApollo(team_id=None), {}
    fake.add_contact("c1", "Ada Lovelace")

    with pytest.raises(ApolloSourceError, match="no team id"):
        _sync(fake, state)

    assert "team_id" not in state
    assert state["fingerprints"] == {}


def test_a_host_can_read_the_team_id_with_one_request():
    fake = _workspace()

    assert apollo_team_id(fake) == "team1"
    assert fake.calls == [("GET", "/contact_stages")]


def test_a_host_asking_for_the_team_id_of_a_stage_list_without_one_gets_an_error():
    with pytest.raises(ApolloSourceError, match="no team id"):
        apollo_team_id(FakeApollo(team_id=None))


# --- rendering is stable whatever order Apollo sends ---


def _shuffled(contact: dict, seed: int) -> dict:
    rng = random.Random(seed)
    contact = dict(contact)
    for key in ("label_ids", "contact_campaign_statuses"):
        contact[key] = rng.sample(contact[key], len(contact[key]))
    fields = list(contact["typed_custom_fields"].items())
    contact["typed_custom_fields"] = dict(rng.sample(fields, len(fields)))
    return contact


def test_a_contact_renders_the_same_whatever_order_apollo_sends():
    fake = FakeApollo()
    contact = fake.add_contact(
        "c1",
        "Ada Lovelace",
        label_ids=["l1", "l2"],
        typed_custom_fields={"f1": "Data quality", "f2": "Staff"},
        contact_campaign_statuses=[
            {"id": "x1", "emailer_campaign_id": "s1", "status": "active"},
            {"id": "x2", "emailer_campaign_id": "s2", "status": "paused"},
        ],
    )
    lookups = Lookups(
        labels={"l1": "Champions", "l2": "Key Accounts"},
        fields={"contact.f1": "Messaging Angle", "contact.f2": "Seniority"},
        sequences={"s1": "Q4 Outreach", "s2": "Renewals"},
    )

    expected = render_contact(contact, lookups)
    for seed in range(20):
        assert render_contact(_shuffled(contact, seed), lookups) == expected


def test_a_multi_select_value_renders_sorted_whatever_order_apollo_sends():
    # An empty choice and a nested list are dropped or flattened the same way each time. Real
    # picklists answer option ids, not labels (apollo.md): text stands in for the choices.
    lookups = Lookups(fields={"contact.f1": "Interests"})
    first = FakeApollo().add_contact("c1", "Ada", typed_custom_fields={"f1": ["b", "", ["d", "c"]]})
    second = FakeApollo().add_contact("c1", "Ada", typed_custom_fields={"f1": [["c", "d"], "b"]})

    assert render_contact(first, lookups) == render_contact(second, lookups)
    assert "Interests: b, c, d" in render_contact(first, lookups)["content"].splitlines()


def test_a_reordered_contact_is_not_re_emitted():
    fake, state = FakeApollo(), {}
    fake.add_sequence("s1", "Q4 Outreach")
    fake.add_sequence("s2", "Renewals")
    fake.add_contact(
        "c1",
        "Ada Lovelace",
        label_ids=["l1", "l2"],
        typed_custom_fields={"f1": "Data quality", "f2": "Staff"},
    )
    fake.enroll("c1", "s1")
    fake.contacts["c1"]["contact_campaign_statuses"].append(
        {"id": "x2", "emailer_campaign_id": "s2", "status": "paused", "current_step_position": 1}
    )
    options = {
        "include": ("contacts",),
        "custom_field_ids": frozenset({"contact.f1", "contact.f2"}),
    }
    _sync(fake, state, **options)

    contact = fake.contacts["c1"]
    for key in ("label_ids", "contact_campaign_statuses"):
        contact[key] = contact[key][::-1]
    contact["typed_custom_fields"] = dict(reversed(contact["typed_custom_fields"].items()))
    rows, stats = _sync(fake, state, **options)

    assert rows == []
    assert stats["skipped"] == 1


def test_a_record_is_checked_once_by_the_run_that_resumes_a_cut_one():
    """A run cut in the middle of forgetting leaves its ids pending. The next run checks them
    first, and the same record comes up again when its listing closes: one run must still
    count one failure, or a record reaches back-off in half the runs."""
    fake, state = _workspace(), {}
    _sync(fake, state)
    fake.hidden.add("c2")
    fake.fail_paths["/contacts/c2"] = ApolloRateLimitedError("Apollo rate limit exceeded")
    _sync(fake, state)
    assert state["pending_deletes"] == ["contact:c2"]
    fake.fail_paths["/contacts/c2"] = ApolloAPIError("Apollo request failed: HTTP 500")
    checks_before = fake.count("/contacts/c2")

    _, stats = _sync(fake, state)

    assert fake.count("/contacts/c2") - checks_before == 1
    assert stats["failed"] == 1
    assert state["check_failures"]["contact:c2"]["failed"] == 1


def test_a_cut_while_forgetting_finishes_the_pending_deletes_next_run():
    fake, state = _workspace(), {}
    _sync(fake, state)
    del fake.contacts["c1"]
    del fake.contacts["c2"]

    fake.fail_paths["/contacts/c2"] = ApolloRateLimitedError("Apollo rate limit exceeded")
    rows, stats = _sync(fake, state)

    assert _tombstones(rows) == {"contact:c1"}
    assert stats["failed_rate_limit"] == 1
    assert state["pending_deletes"] == ["contact:c2"]

    del fake.fail_paths["/contacts/c2"]
    rows, _ = _sync(fake, state)

    assert _tombstones(rows) == {"contact:c2"}
