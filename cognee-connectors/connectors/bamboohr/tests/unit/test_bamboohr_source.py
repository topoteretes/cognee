"""The walk: fingerprints, forget-on-delete, budgets, and every count a run reports.

The walk is driven with a plain dict as dlt resource state and the in-memory
FakeBambooHR: no network, no dlt, no LLM.
"""

import pytest
from fake_bamboohr import FakeBambooHR

from cognee_connector_bamboohr.source import (
    BambooHRAccessError,
    BambooHRAPIError,
    BambooHRAuthError,
    BambooHRSourceError,
    _iter_rows,
    new_stats,
)
from cognee_connectors_core.dlt_conventions import NODE_SET_COLUMN

FIELDS = ["department", "firstName", "jobTitle", "lastName", "location", "status"]
COUNT_KEYS = {"scanned", "skipped", "deleted", "failed", "kept", "pending", "requests", "held_back"}
ADA = {
    "firstName": "Ada",
    "lastName": "Lovelace",
    "jobTitle": "Engineer",
    "department": "Research",
    "location": "London",
    "status": "Active",
}
GRACE = {
    "firstName": "Grace",
    "lastName": "Hopper",
    "jobTitle": "Engineer",
    "department": "Research",
    "location": "Arlington",
    "status": "Active",
}


def _sync(fake, state=None, fields=FIELDS, max_requests=500, **kwargs):
    state = {} if state is None else state
    stats = new_stats()
    rows = list(
        _iter_rows(
            fake,
            state,
            stats,
            subdomain="acme",
            fields=fields,
            max_requests=max_requests,
            **kwargs,
        )
    )
    return rows, state, stats


def _ids(rows):
    return {row["id"] for row in rows}


def _tombstones(rows):
    return {row["id"] for row in rows if row.get("_deleted")}


def _row(rows, row_id):
    return next(row for row in rows if row["id"] == row_id)


def _company(**employees):
    return FakeBambooHR({"7": dict(ADA), "9": dict(GRACE), **employees})


def test_the_first_sync_emits_every_employee_with_the_frozen_row_shape():
    fake = _company()
    rows, state, stats = _sync(fake)

    assert _ids(rows) == {"employee:7", "employee:9"}
    row = _row(rows, "employee:7")
    assert row["title"] == "Ada Lovelace"
    assert row["content"].startswith("BambooHR employee Ada Lovelace.")
    assert "Department: Research" in row["content"]
    assert row["url"] == "https://acme.bamboohr.com/employees/view/7"
    assert row[NODE_SET_COLUMN] == ["bamboohr:acme"]
    assert set(row) == {"id", "title", "content", "url", "_deleted", NODE_SET_COLUMN}

    assert stats["scanned"] == 2
    assert stats["requests"] == 2  # the field list and the one report
    assert stats["deleted"] == 0
    assert stats["failed"] == 0
    assert set(state["fingerprints"]) == {"employee:7", "employee:9"}
    assert state["subdomain"] == "acme"


def test_every_request_asks_for_exactly_the_allowlisted_fields():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")
    _sync(fake, state)

    report = next(call for call in fake.request_log if call["path"] == "/employees/custom-report")
    assert report["body"] == {"fields": FIELDS}
    checks = [
        call
        for call in fake.request_log
        if call["path"].startswith("/employees/") and call["path"] != "/employees/custom-report"
    ]
    assert len(checks) == 1
    assert checks[0]["method"] == "GET"
    assert checks[0]["params"] == {"fields": ",".join(FIELDS)}


def test_an_unchanged_resync_emits_nothing():
    fake = _company()
    _, state, _ = _sync(fake)
    rows, _, stats = _sync(fake, state)

    assert rows == []
    assert stats["scanned"] == 2
    assert stats["skipped"] == 2
    assert stats["requests"] == 2


def test_an_edit_re_emits_only_that_employee():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.employees["7"]["department"] = "Research II"

    rows, _, stats = _sync(fake, state)

    assert _ids(rows) == {"employee:7"}
    assert "Department: Research II" in _row(rows, "employee:7")["content"]
    assert stats["skipped"] == 1


def test_changing_the_allowlist_re_renders_the_employees_who_gain_a_line():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.employees["7"]["customTShirtSize"] = "M"

    rows, _, stats = _sync(fake, state, fields=FIELDS + ["customTShirtSize"])

    # Employee 7 gained a line and re-renders; employee 9 has no value for the new
    # field, so the new allowlist alone changes no text and 9 is not re-extracted.
    assert _ids(rows) == {"employee:7"}
    assert "T-shirt size: M" in _row(rows, "employee:7")["content"]
    assert stats["skipped"] == 1


def test_a_deleted_employee_is_tombstoned_after_one_check():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")

    rows, state, stats = _sync(fake, state)

    assert _tombstones(rows) == {"employee:9"}
    assert stats["deleted"] == 1
    assert stats["requests"] == 3  # the field list, the report, the one check
    assert "employee:9" not in state["fingerprints"]
    assert state["pending_deletes"] == []


def test_an_employee_the_report_misses_but_the_api_serves_is_kept():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.hidden.add("9")

    rows, _, stats = _sync(fake, state)

    assert rows == []
    assert _tombstones(rows) == set()
    assert stats["deleted"] == 0
    assert stats["failed"] == 0
    assert stats["requests"] == 3


def test_a_failed_check_keeps_the_employee_and_backs_off_after_five_failures():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.hidden.add("9")  # leaves the report; without the failure the API still serves it
    fake.failures["/employees/9/"] = BambooHRAPIError(500, "/employees/{id}")

    for _ in range(5):
        rows, state, stats = _sync(fake, state)
        assert rows == []
        assert stats["failed"] == 1
        assert stats["failed_deletion_check"] == 1
        assert stats["requests"] == 3  # the check was made and failed

    # The next cycle leaves the stuck employee alone: no check request at all.
    rows, state, stats = _sync(fake, state)
    assert rows == []
    assert stats["held_back"] == 1
    assert stats["failed"] == 0
    assert stats["requests"] == 2  # only the field list and the report

    # After ten quiet cycles the check is tried once more; a record listed again
    # settles without any request.
    for _ in range(9):
        _sync(fake, state)
    fake.hidden.discard("9")
    del fake.failures["/employees/9/"]
    rows, state, stats = _sync(fake, state)
    assert rows == []
    assert stats["held_back"] == 0
    assert stats["requests"] == 2
    assert state["check_failures"] == {}


def test_a_check_refused_with_403_keeps_the_employee_and_counts_the_refusal():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")
    fake.failures["/employees/9/"] = BambooHRAccessError("/employees/{id}")

    rows, _, stats = _sync(fake, state)

    assert _tombstones(rows) == set()
    assert stats["failed"] == 1
    assert stats["failed_access"] == 1
    assert stats["deleted"] == 0


def test_a_field_bamboohr_does_not_list_fails_the_run_before_any_row():
    fake = _company()
    state, stats = {}, new_stats()
    with pytest.raises(BambooHRSourceError, match="noSuchField") as raised:
        list(_iter_rows(fake, state, stats, subdomain="acme", fields=FIELDS + ["noSuchField"]))

    assert "noSuchField" in str(raised.value)
    assert stats["failed"] == 1
    assert state["fingerprints"] == {}  # nothing emitted, nothing to forget against
    assert fake.calls == [("GET", "/meta/fields")]  # the report was never asked


def test_a_report_the_key_cannot_call_raises_and_forgets_nothing():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")
    fake.failures["/employees/custom-report"] = BambooHRAccessError("/employees/custom-report")

    with pytest.raises(BambooHRAccessError):
        _sync(fake, state)

    assert state["pending_deletes"] == []  # the cut listing forgets nothing
    assert "employee:9" in state["fingerprints"]


def test_another_companys_state_under_one_resource_name_is_refused():
    fake = _company()
    with pytest.raises(BambooHRSourceError, match="another BambooHR company"):
        _sync(fake, state={"subdomain": "other"})


def test_a_spent_budget_stops_the_run_cleanly_and_the_next_resumes_it():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")

    # Two requests cover the field list and the report; the check cannot be afforded.
    rows, state, stats = _sync(fake, state, max_requests=2)
    assert _tombstones(rows) == set()
    assert stats["failed"] == 1
    assert stats["failed_budget"] == 1
    assert stats["requests"] == 2
    assert COUNT_KEYS <= set(stats)
    assert state["pending_deletes"] == ["employee:9"]  # kept for the next run

    rows, state, stats = _sync(fake, state, max_requests=3)
    assert _tombstones(rows) == {"employee:9"}
    assert stats["deleted"] == 1
    assert stats["failed"] == 0
    assert state["pending_deletes"] == []


def test_a_rate_limit_stops_the_run_cleanly_and_keeps_the_state():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")
    fake.rate_limited = True

    rows, state, stats = _sync(fake, state)

    assert rows == []
    assert stats["failed"] == 1
    assert stats["failed_rate_limit"] == 1
    assert state["pending_deletes"] == []
    assert "employee:9" in state["fingerprints"]


def test_a_key_rejected_before_any_row_raises_and_drops_the_run():
    fake = _company()
    fake.failures["/meta/fields"] = BambooHRAuthError("/meta/fields")

    with pytest.raises(BambooHRAuthError):
        _sync(fake)

    assert fake.calls == [("GET", "/meta/fields")]


def test_a_key_rejected_after_rows_stops_the_run_and_the_next_resumes_it():
    fake = _company()
    _, state, _ = _sync(fake)
    fake.remove_employee("9")
    fake.employees["7"]["department"] = "Research II"  # one row to emit before the cut
    fake.failures["/employees/9/"] = BambooHRAuthError("/employees/{id}")

    rows, state, stats = _sync(fake, state)
    assert _ids(rows) == {"employee:7"}  # the emitted row is kept...
    assert _tombstones(rows) == set()  # ...but the refused check forgets nothing
    assert stats["failed"] == 1
    assert stats["failed_auth"] == 1
    assert state["pending_deletes"] == ["employee:9"]

    del fake.failures["/employees/9/"]
    rows, state, stats = _sync(fake, state)
    assert _tombstones(rows) == {"employee:9"}
    assert stats["deleted"] == 1


def test_a_check_active_stop_propagates_and_emits_nothing():
    fake = _company()

    def revoked():
        raise RuntimeError("the host revoked the run")

    with pytest.raises(RuntimeError, match="revoked"):
        _sync(fake, check_active=revoked)
    assert fake.calls == []


def test_an_employee_id_with_characters_bamboohr_could_not_have_sent_is_never_requested():
    fake = _company()
    rows, state, stats = _sync(
        fake,
        state={
            "subdomain": "acme",
            "fingerprints": {"employee:..": "x"},
            "seen": [],
            "pending_deletes": [],
        },
    )

    assert _tombstones(rows) == set()
    assert stats["failed"] == 1
    assert stats["failed_deletion_check"] == 1
    assert "employee:.." in state["fingerprints"]  # kept, not requested
    assert not any(".." in call["path"] for call in fake.request_log)
