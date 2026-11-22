# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The Evernote walk against a fake account: what each run emits, deletes and keeps in state."""

import copy
import json
import logging
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.connectors.evernote import (
    CHUNK_SIZE,
    EvernoteAuthError,
    EvernoteError,
    EvernotePermissionError,
    EvernoteRateLimitError,
    EvernoteUnavailableError,
    _iter_rows,
    evernote_source,
    new_stats,
)
from cognee.tests.unit.tasks.connectors.fake_evernote import FakeEvernote, account

STATS_KEYS = {
    "emitted",
    "deleted",
    "failed",
    "failed_rate_limit",
    "failed_auth",
    "failed_unavailable",
}

pytestmark = pytest.mark.timeout(60)


def _sync(fake, state, **kwargs):
    stats = new_stats()
    rows = list(_iter_rows(fake, state, stats, **kwargs))
    return rows, stats


def _ids(rows, deleted=False):
    """Note guids of the rows, after checking each row id is ``note:<guid>``."""
    ids = [row["id"] for row in rows if row["_deleted"] is deleted]
    assert all(row_id.startswith("note:") for row_id in ids), ids
    return [row_id.removeprefix("note:") for row_id in ids]


def _rejected():
    return EvernoteAuthError("Evernote rejected the token (AUTH_EXPIRED (9))")


def _limited():
    return EvernoteRateLimitError("Evernote rate limit reached, retry after 900s", 900)


# -- first sync and incremental runs -------------------------------------------------------------


def test_first_sync_emits_every_note_and_records_where_it_stopped():
    fake = account(3)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _ids(rows) == ["n1", "n2", "n3"]
    assert rows[0] == {
        "id": "note:n1",
        "title": "Note n1",
        "content": "Notebook: Work\n\nBody of n1",
        "_deleted": False,
    }
    assert stats == {**new_stats(), "emitted": 3}
    assert fake.calls_to("get_filtered_sync_chunk")[0] == (0, CHUNK_SIZE)
    assert state["last_usn"] == fake.update_count
    assert state["last_sync_time"] == fake.current_time
    assert state["account"] == "production:1001"
    assert state["note_ids"] == {"n1": "nb1", "n2": "nb1", "n3": "nb1"}
    assert "listing" not in state


def test_a_second_run_with_nothing_changed_reads_no_chunk_and_leaves_state_as_it_was():
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    before = copy.deepcopy(state)
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert rows == [] and stats == new_stats()
    assert fake.calls_to("get_filtered_sync_chunk") == []
    assert state == before


def test_an_edit_between_runs_is_emitted_again_and_nothing_else_is_fetched():
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    fake.calls.clear()
    fake.edit_note("n2", "Rewritten body")

    rows, stats = _sync(fake, state)

    assert [(r["id"], r["content"]) for r in rows] == [
        ("note:n2", "Notebook: Work\n\nRewritten body")
    ]
    assert fake.calls_to("get_note") == [("n2",)]
    assert fake.calls_to("get_filtered_sync_chunk")[0][0] > 0
    assert stats["emitted"] == 1 and stats["deleted"] == 0


def test_a_listing_longer_than_one_chunk_reads_every_chunk_in_order():
    fake = account(CHUNK_SIZE + 10)
    state: dict = {}

    rows, _ = _sync(fake, state)

    assert len(_ids(rows)) == CHUNK_SIZE + 10
    starts = [call[0] for call in fake.calls_to("get_filtered_sync_chunk")]
    assert starts[0] == 0 and starts[1] == CHUNK_SIZE and starts == sorted(starts)


def test_a_chunk_that_does_not_advance_raises_instead_of_looping():
    fake = account(2)
    original = fake.get_filtered_sync_chunk

    def stuck(after_usn, max_entries, check_active=None):
        chunk = original(after_usn, max_entries, check_active)
        chunk.chunkHighUSN = after_usn
        chunk.updateCount = after_usn + 100
        return chunk

    fake.get_filtered_sync_chunk = stuck
    state: dict = {}
    stats = new_stats()

    with pytest.raises(EvernoteError, match="did not advance"):
        list(_iter_rows(fake, state, stats))
    assert stats["failed"] == 1 and state == {}


# -- deletion ------------------------------------------------------------------------------------


def test_trashed_and_expunged_notes_become_tombstones():
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    fake.trash_note("n1")
    fake.expunge_note("n2")

    rows, stats = _sync(fake, state)

    assert sorted(_ids(rows, deleted=True)) == ["n1", "n2"]
    assert stats["deleted"] == 2
    assert state["note_ids"] == {"n3": "nb1"}


def test_an_expunged_notebook_deletes_the_notes_that_were_in_it():
    fake = account(2)
    fake.add_note("h1", "Home note", notebook="nb2")
    state: dict = {}
    _sync(fake, state)

    fake.expunge_notebook("nb2")
    rows, _ = _sync(fake, state)

    assert _ids(rows, deleted=True) == ["h1"]
    assert "nb2" not in state["notebooks"]
    assert set(state["note_ids"]) == {"n1", "n2"}


def test_a_note_moved_out_of_an_expunged_notebook_first_is_kept():
    fake = account(1)
    fake.add_note("h1", "Home note", notebook="nb2")
    state: dict = {}
    _sync(fake, state)

    fake.edit_note("h1", notebook="nb1")
    fake.expunge_notebook("nb2")
    rows, _ = _sync(fake, state)

    assert _ids(rows, deleted=True) == []
    assert "h1" in _ids(rows) and state["note_ids"]["h1"] == "nb1"


def test_only_notes_that_were_emitted_get_a_tombstone():
    fake = account(1)
    fake.add_note("trashed", "never synced")
    fake.trash_note("trashed")
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _ids(rows, deleted=True) == [] and stats["deleted"] == 0


def test_a_note_created_and_trashed_between_runs_gets_no_tombstone():
    fake = account(1)
    state: dict = {}
    _sync(fake, state)
    fake.add_note("brief", "never synced")
    fake.trash_note("brief")

    rows, stats = _sync(fake, state)

    assert rows == [] and stats["deleted"] == 0


def test_a_note_evernote_no_longer_finds_is_deleted():
    """Listed as changed, gone before its body is read: the 404 of Thrift is the evidence."""
    from cognee.tasks.ingestion.connectors.evernote import NoteNotFoundError

    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_note("n1", "edited")
    fake.fail("get_note", NoteNotFoundError("Evernote note not found"))

    rows, stats = _sync(fake, state)

    assert _ids(rows, deleted=True) == ["n1"] and stats["deleted"] == 1
    assert "n1" not in state["note_ids"]


def test_an_empty_note_without_a_title_is_deleted_and_with_a_title_keeps_its_title():
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.notes["n1"]["content"] = "<en-note></en-note>"
    fake.edit_note("n1", title="")
    fake.notes["n2"]["content"] = "<en-note></en-note>"
    fake.edit_note("n2")

    rows, _ = _sync(fake, state)

    assert _ids(rows, deleted=True) == ["n1"]
    assert [r["content"] for r in rows if r["id"] == "note:n2"] == ["Notebook: Work\n\nNote n2"]


def test_full_sync_before_newer_than_the_last_sync_lists_the_whole_account_again():
    """Evernote's own signal that ``last_usn`` cannot be trusted: the run lists from USN 0 and
    forgets what the complete listing no longer holds, even without an expunge record."""
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    fake.drop_note_silently("n2")
    fake.full_sync_before = state["last_sync_time"] + 1
    fake.calls.clear()

    rows, _ = _sync(fake, state)

    assert fake.calls_to("get_filtered_sync_chunk")[0][0] == 0
    assert _ids(rows, deleted=True) == ["n2"]
    assert set(state["note_ids"]) == {"n1", "n3"}


def test_an_update_count_below_the_stored_one_lists_the_whole_account_again():
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    state["last_usn"] = fake.update_count + 50
    fake.calls.clear()

    _sync(fake, state)

    assert fake.calls_to("get_filtered_sync_chunk")[0][0] == 0
    assert state["last_usn"] == fake.update_count


def test_full_resync_forgets_notes_that_are_gone_and_keeps_the_rest():
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    fake.drop_note_silently("n3")

    rows, _ = _sync(fake, state, full_resync=True)

    assert _ids(rows, deleted=True) == ["n3"]
    assert sorted(_ids(rows)) == ["n1", "n2"]


# -- selection -----------------------------------------------------------------------------------


def test_a_notebook_selection_emits_only_its_notes():
    fake = account(1)
    fake.add_note("h1", "Home note", notebook="nb2")
    state: dict = {}

    rows, _ = _sync(fake, state, notebook_names=["Home"])

    assert _ids(rows) == ["h1"]
    assert state["scope"]["notebook_guids"] == ["nb2"]


def test_a_tag_selection_emits_only_tagged_notes_and_renders_tags_sorted():
    fake = account(0)
    fake.add_note("a", "tagged", tags=["t2", "t1"])
    fake.add_note("b", "untagged")
    state: dict = {}

    rows, _ = _sync(fake, state, tag_names=["urgent"])

    assert _ids(rows) == ["a"]
    assert rows[0]["content"] == "Notebook: Work\nTags: later, urgent\n\ntagged"


def test_a_note_moved_out_of_the_selection_is_deleted():
    fake = account(2)
    state: dict = {}
    _sync(fake, state, notebook_names=["Work"])
    fake.edit_note("n1", notebook="nb2")

    rows, _ = _sync(fake, state, notebook_names=["Work"])

    assert _ids(rows, deleted=True) == ["n1"]


def test_narrowing_the_selection_lists_again_and_forgets_what_left_it():
    fake = account(1)
    fake.add_note("h1", "Home note", notebook="nb2")
    state: dict = {}
    _sync(fake, state)
    fake.calls.clear()

    rows, _ = _sync(fake, state, notebook_names=["Work"])

    assert fake.calls_to("get_filtered_sync_chunk")[0][0] == 0
    assert _ids(rows, deleted=True) == ["h1"]


def test_an_unknown_notebook_name_fails_the_run_before_any_row_and_keeps_state():
    fake = account(2)
    state: dict = {}
    _sync(fake, state, notebook_names=["Work"])
    before = copy.deepcopy(state)

    with pytest.raises(EvernoteError, match=r"Notebook name\(s\) not found.*\['Wrok'\]"):
        _sync(fake, state, notebook_names=["Wrok"])
    assert state == before


def test_renaming_a_selected_notebook_fails_the_run_with_a_clear_message():
    """Known limitation: the selection is by name, so the rename has to reach it too."""
    fake = account(2)
    state: dict = {}
    _sync(fake, state, notebook_names=["Work"])
    before = copy.deepcopy(state)
    fake.rename_notebook("nb1", "Job")

    stats = new_stats()
    with pytest.raises(EvernoteError, match="renamed notebook must be renamed in the selection"):
        list(_iter_rows(fake, state, stats, notebook_names=["Work"]))
    assert state == before and stats["failed"] == 1


def test_renaming_a_selected_tag_fails_the_run_with_a_clear_message():
    fake = account(0)
    fake.add_note("a", "tagged", tags=["t1"])
    state: dict = {}
    _sync(fake, state, tag_names=["urgent"])
    fake.rename_tag("t1", "asap")

    with pytest.raises(EvernoteError, match="renamed tag must be renamed in the selection"):
        _sync(fake, state, tag_names=["urgent"])


def test_a_selected_name_moving_to_another_notebook_lists_again_and_swaps_the_notes():
    fake = account(1)
    fake.add_note("h1", "Home note", notebook="nb2")
    state: dict = {}
    _sync(fake, state, notebook_names=["Work"])
    fake.rename_notebook("nb1", "Old work")
    fake.rename_notebook("nb2", "Work")

    rows, _ = _sync(fake, state, notebook_names=["Work"])

    assert _ids(rows, deleted=True) == ["n1"] and _ids(rows) == ["h1"]
    assert state["scope"]["notebook_guids"] == ["nb2"]


def test_a_renamed_notebook_keeps_its_old_name_in_a_note_until_the_note_changes():
    """Known limitation: the name is part of the text, and only a changed note is fetched."""
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.rename_notebook("nb1", "Job")

    rows, _ = _sync(fake, state)
    assert rows == []

    fake.edit_note("n1", "Body of n1")
    rows, _ = _sync(fake, state)
    assert rows[0]["content"].startswith("Notebook: Job\n")


@pytest.mark.parametrize("names", ["Work", ["Work", ""], [1]])
def test_a_selection_that_is_not_a_list_of_names_is_refused(names):
    with pytest.raises(ValueError, match="notebook_names must be a list or tuple"):
        evernote_source(resource_name="evernote_x", service=account(), notebook_names=names)


def test_the_same_selection_in_another_order_is_the_same_selection():
    fake = account(0)
    fake.add_note("a", "x", tags=["t1"])
    first = evernote_source(resource_name="e", service=fake, tag_names=["urgent", "later"])
    second = evernote_source(resource_name="e", service=fake, tag_names=["later", "urgent"])
    state: dict = {}
    list(_iter_rows(fake, state, new_stats(), tag_names=["later", "urgent"]))
    fake.calls.clear()

    list(_iter_rows(fake, state, new_stats(), tag_names=["later", "urgent"]))

    assert fake.calls_to("get_filtered_sync_chunk") == []
    assert first.name == second.name


def test_tag_order_from_evernote_does_not_change_the_document():
    def render(order):
        fake = account(0)
        fake.add_note("a", "x", tags=order)
        rows, _ = _sync(fake, {})
        return rows[0]

    tags = ["t1", "t2"]
    assert render(tags) == render(list(reversed(tags)))


# -- EV1: a cut metadata pass continues where it stopped -----------------------------------------


def test_a_rate_limit_inside_the_first_metadata_pass_keeps_progress_and_the_next_run_continues():
    """EV1. The community connector restarted pass 1 from USN 0 on every run and kept nothing
    when the rate limit hit it, so an account larger than one run's quota never moved."""
    fake = account(CHUNK_SIZE * 2 + 5)
    fake.fail("get_filtered_sync_chunk", _limited(), call=2)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert rows == []
    assert stats == {**new_stats(), "failed": 1, "failed_rate_limit": 1}
    assert state["listing"]["usn"] == CHUNK_SIZE
    assert "last_usn" not in state
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    starts = [call[0] for call in fake.calls_to("get_filtered_sync_chunk")]
    assert starts[0] == CHUNK_SIZE, "the second run continues, it does not start from USN 0"
    assert len(_ids(rows)) == CHUNK_SIZE * 2 + 5, "every note, once"
    assert stats["failed"] == 0 and "listing" not in state
    assert state["last_usn"] == fake.update_count


def test_a_cut_metadata_pass_never_deletes_and_the_completed_one_does():
    fake = account(CHUNK_SIZE + 5)
    state: dict = {}
    _sync(fake, state)
    fake.drop_note_silently("n3")
    fake.fail("get_filtered_sync_chunk", _limited(), call=2)

    rows, _ = _sync(fake, state, full_resync=True)

    assert rows == [] and "n3" in state["note_ids"], "no tombstone from a partial listing"
    rows, _ = _sync(fake, state, full_resync=True)
    assert _ids(rows, deleted=True) == ["n3"]


def test_a_saved_metadata_pass_older_than_full_sync_before_starts_over():
    fake = account(CHUNK_SIZE + 5)
    fake.fail("get_filtered_sync_chunk", _limited(), call=2)
    state: dict = {}
    _sync(fake, state)
    fake.full_sync_before = state["listing"]["started"] + 1
    fake.calls.clear()

    rows, _ = _sync(fake, state)

    assert fake.calls_to("get_filtered_sync_chunk")[0][0] == 0
    assert len(_ids(rows)) == CHUNK_SIZE + 5


def test_a_saved_metadata_pass_for_another_selection_starts_over():
    fake = account(CHUNK_SIZE + 5)
    fake.fail("get_filtered_sync_chunk", _limited(), call=2)
    state: dict = {}
    _sync(fake, state)
    fake.calls.clear()

    _sync(fake, state, notebook_names=["Work"])

    assert fake.calls_to("get_filtered_sync_chunk")[0][0] == 0


# -- cut and resume in the body pass -------------------------------------------------------------


def test_a_rate_limit_on_the_seventh_body_keeps_six_and_the_next_run_fetches_only_the_rest():
    fake = account(10)
    fake.fail("get_note", _limited(), call=7)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _ids(rows) == [f"n{i}" for i in range(1, 7)]
    assert stats == {**new_stats(), "emitted": 6, "failed": 1, "failed_rate_limit": 1}
    assert "last_usn" not in state, "the sync is not complete"
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert fake.calls_to("get_filtered_sync_chunk") == [], "pass 1 is not read again"
    assert [c[0] for c in fake.calls_to("get_note")] == ["n7", "n8", "n9", "n10"]
    assert stats["failed"] == 0 and state["last_usn"] == fake.update_count


def test_max_notes_per_run_spreads_a_first_sync_over_runs_and_loses_nothing():
    fake = account(10)
    state: dict = {}
    seen: list[str] = []
    for _ in range(3):
        rows, stats = _sync(fake, state, max_notes_per_run=4)
        seen += _ids(rows)
        assert stats["failed"] == 0
    assert seen == [f"n{i}" for i in range(1, 11)]
    assert "listing" not in state


def test_an_edit_to_a_note_already_emitted_in_a_cut_sync_is_emitted_again():
    fake = account(4)
    state: dict = {}
    _sync(fake, state, max_notes_per_run=2)
    fake.edit_note("n1", "Edited after it was emitted")
    _sync(fake, state, max_notes_per_run=2)  # finishes n3, n4

    rows, _ = _sync(fake, state)

    assert [(r["id"], r["content"].split("\n\n")[1]) for r in rows] == [
        ("note:n1", "Edited after it was emitted")
    ]


def test_check_active_is_called_before_every_request():
    fake = account(2)
    checks: list[int] = []

    rows, _ = _sync(fake, {}, check_active=lambda: checks.append(len(fake.calls)))

    assert len(checks) == len(fake.calls)
    assert len(rows) == 2


def test_what_check_active_raises_stops_the_run_unwrapped_and_keeps_state():
    class Revoked(Exception):
        pass

    fake = account(3)
    state: dict = {}

    def check():
        if len(fake.calls_to("get_note")) == 1:
            raise Revoked

    stats = new_stats()
    with pytest.raises(Revoked):
        list(_iter_rows(fake, state, stats, check_active=check))
    assert state == {} and stats["failed"] == 0


def test_a_raised_run_leaves_state_as_it_was():
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    before = copy.deepcopy(state)
    fake.edit_note("n2", "edited")
    fake.fail("get_sync_state", EvernoteError("Evernote system error INTERNAL_ERROR (4)"))

    stats = new_stats()
    with pytest.raises(EvernoteError):
        list(_iter_rows(fake, state, stats))
    assert state == before and stats["failed"] == 1


# -- a note whose body fails is retried, never blocking the sync --------------------------------


def _broken():
    """A note this token may not read: the per-note failure."""
    from cognee.tasks.ingestion.connectors.evernote import EvernotePermissionError

    return EvernotePermissionError("Evernote denied access (PERMISSION_DENIED (3))")


def test_a_note_whose_body_fails_is_skipped_counted_and_fetched_again_next_run():
    """The community connector raised here, so one note Evernote could not serve held every
    later run at the same place."""
    fake = account(3)
    fake.fail("get_note", _broken(), call=2)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _ids(rows) == ["n1", "n3"]
    assert stats == {**new_stats(), "emitted": 2, "failed": 1}
    assert state["last_usn"] == fake.update_count, "the sync moved on"
    assert list(state["retry"]) == ["n2"]
    fake.calls.clear()

    rows, stats = _sync(fake, state)

    assert fake.calls_to("get_note") == [("n2",)]
    assert _ids(rows) == ["n2"] and stats["failed"] == 0
    assert "retry" not in state


def test_a_note_that_keeps_failing_stays_in_retry_and_its_old_document_stays():
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_note("n1", "edited")
    fake.fail("get_note", _broken(), times=2)

    for _ in range(2):
        rows, stats = _sync(fake, state)
        assert rows == [] and stats["failed"] == 1
        assert list(state["retry"]) == ["n1"] and "n1" in state["note_ids"]


@pytest.mark.parametrize("how", ["expunge_note", "expunge_notebook"])
def test_a_retried_note_gone_at_evernote_gets_its_tombstone_without_a_fetch(how):
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.edit_note("n1", "edited")
    fake.fail("get_note", _broken())
    _sync(fake, state)
    getattr(fake, how)("n1" if how == "expunge_note" else "nb1")
    fake.calls.clear()

    rows, _ = _sync(fake, state)

    assert "n1" in _ids(rows, deleted=True)
    assert ("n1",) not in fake.calls_to("get_note")
    assert "retry" not in state


def test_a_retried_note_never_emitted_and_then_not_found_gets_no_tombstone():
    from cognee.tasks.ingestion.connectors.evernote import NoteNotFoundError

    fake = account(1)
    fake.fail("get_note", _broken())
    state: dict = {}
    _sync(fake, state)
    fake.fail("get_note", NoteNotFoundError("Evernote note not found"))

    rows, stats = _sync(fake, state)

    assert rows == [] and stats["deleted"] == 0 and "retry" not in state


def test_an_edit_to_a_note_in_retry_is_fetched_with_its_new_metadata():
    fake = account(1)
    fake.fail("get_note", _broken())
    state: dict = {}
    _sync(fake, state)
    fake.edit_note("n1", "edited", title="Renamed")

    rows, _ = _sync(fake, state)

    assert [r["title"] for r in rows] == ["Renamed"] and "retry" not in state


def test_a_note_that_keeps_being_denied_goes_to_retry_while_the_others_sync():
    """PERMISSION_DENIED on one body is about that note; first in USN order it used to stop
    every run before any other note."""
    from cognee.tasks.ingestion.connectors.evernote import EvernotePermissionError

    fake = account(3)
    fake.fail("get_note", EvernotePermissionError("Evernote denied access"), times=2, guid="n1")
    state: dict = {}

    rows, stats = _sync(fake, state)
    assert _ids(rows) == ["n2", "n3"]
    assert stats == {**new_stats(), "emitted": 2, "failed": 1}
    rows, stats = _sync(fake, state)
    assert rows == [] and stats["failed"] == 1 and list(state["retry"]) == ["n1"]
    rows, _ = _sync(fake, state)
    assert _ids(rows) == ["n1"] and "retry" not in state


def test_a_change_outside_the_filter_does_not_make_every_idle_run_list():
    """A saved search or a resource takes a USN that no chunk of this filter returns."""
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.update_count += 1  # e.g. a saved search was edited

    _sync(fake, state)
    assert state["last_usn"] == fake.update_count
    fake.calls.clear()
    rows, _ = _sync(fake, state)

    assert rows == [] and fake.calls_to("get_filtered_sync_chunk") == []


def test_a_note_changed_after_an_ignored_change_is_still_read():
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.update_count += 1
    fake.edit_note("n1", "edited")

    rows, _ = _sync(fake, state)

    assert _ids(rows) == ["n1"]


# -- a rejected token in every phase (Cloud readiness row 3) -------------------------------------
#
# Evernote answers a bad token with EDAMUserException INVALID_AUTH or AUTH_EXPIRED inside an HTTP
# 200, never an HTTP 401; the client maps both to EvernoteAuthError.


@pytest.mark.parametrize(
    "method", ["get_user_id", "get_sync_state", "get_filtered_sync_chunk", "get_note"]
)
def test_a_rejected_token_before_any_row_raises_an_auth_error_and_keeps_state(method):
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    fake.edit_note("n1", "edited")
    before = copy.deepcopy(state)
    fake.fail(method, _rejected())

    stats = new_stats()
    with pytest.raises(EvernoteAuthError):
        list(_iter_rows(fake, state, stats))
    assert state == before
    assert stats == {**new_stats(), "failed": 1, "failed_auth": 1}


def test_a_rejected_token_after_rows_stops_cleanly_with_failed_auth_and_the_next_run_resumes():
    """The 401 of Thrift after rows: what was emitted stays, the rest follows next run."""
    fake = account(5)
    fake.fail("get_note", _rejected(), call=3)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _ids(rows) == ["n1", "n2"]
    assert stats == {**new_stats(), "emitted": 2, "failed": 1, "failed_auth": 1}
    assert set(state["note_ids"]) == {"n1", "n2"}
    rows, stats = _sync(fake, state)
    assert _ids(rows) == ["n3", "n4", "n5"] and stats["failed"] == 0


def test_a_rejected_token_after_tombstones_stops_cleanly_too():
    fake = account(3)
    state: dict = {}
    _sync(fake, state)
    fake.trash_note("n1")
    fake.edit_note("n2", "edited")
    fake.fail("get_note", _rejected())

    rows, stats = _sync(fake, state)

    assert _ids(rows, deleted=True) == ["n1"] and stats["failed_auth"] == 1
    assert "n1" not in state["note_ids"] and state["listing"]["pending"][0]["guid"] == "n2"


# -- accounts, stats, construction ---------------------------------------------------------------


def test_a_state_bound_to_another_account_raises_and_stays_untouched():
    state: dict = {}
    _sync(account(1, user_id=1), state)
    before = copy.deepcopy(state)

    stats = new_stats()
    with pytest.raises(EvernoteError, match="bound to another Evernote account"):
        list(_iter_rows(account(1, user_id=2), state, stats))
    assert state == before and stats["failed"] == 1


def test_sandbox_and_production_of_one_user_id_are_different_accounts():
    state: dict = {}
    _sync(account(1), state, sandbox=True)
    with pytest.raises(EvernoteError, match="bound to another"):
        _sync(account(1), state)


@pytest.mark.parametrize(
    "setup",
    [
        "clean",
        "rate_limit_listing",
        "rate_limit_body",
        "auth_first",
        "auth_later",
        "cap",
        "note_error",
        "error",
    ],
)
def test_every_stats_key_exists_on_every_exit(setup):
    fake = account(4)
    kwargs = {}
    if setup == "rate_limit_listing":
        fake.fail("get_filtered_sync_chunk", _limited())
    elif setup == "rate_limit_body":
        fake.fail("get_note", _limited(), call=2)
    elif setup == "auth_first":
        fake.fail("get_user_id", _rejected())
    elif setup == "auth_later":
        fake.fail("get_note", _rejected(), call=2)
    elif setup == "cap":
        kwargs["max_notes_per_run"] = 1
    elif setup == "note_error":
        fake.fail("get_note", EvernoteError("Evernote system error"))
    elif setup == "error":
        fake.fail("get_sync_state", EvernoteError("Evernote system error"))
    stats = new_stats()
    try:
        list(_iter_rows(fake, {}, stats, **kwargs))
    except EvernoteError:
        pass
    assert set(stats) == STATS_KEYS
    assert all(type(value) is int and value >= 0 for value in stats.values())


def test_no_guid_reaches_a_log_line(caplog):
    fake = account(3)
    fake.add_note("secretguid-77", "x")
    fake.fail("get_note", _limited(), call=4)

    with caplog.at_level(logging.DEBUG):
        _sync(fake, {})
        _sync(account(3), {}, max_notes_per_run=1)

    assert "secretguid-77" not in caplog.text and "n1" not in caplog.text


def test_construction_validates_and_makes_no_call():
    fake = FakeEvernote()
    source = evernote_source(resource_name="evernote_notes", service=fake)
    assert fake.calls == []
    assert source.name == "evernote_notes"
    assert source.cognee_sync_stats == new_stats()
    with pytest.raises(ValueError, match="resource_name"):
        evernote_source(resource_name=" ", service=fake)
    with pytest.raises(ValueError, match="_dlt_"):
        evernote_source(resource_name="_dlt_loads", service=fake)
    with pytest.raises(ValueError, match="service or a non-empty token"):
        evernote_source(resource_name="e", token="  ")
    with pytest.raises(ValueError, match="max_notes_per_run"):
        evernote_source(resource_name="e", service=fake, max_notes_per_run=0)
    with pytest.raises(TypeError):
        evernote_source("evernote_notes", service=fake)  # keywords only


def test_the_resource_is_a_merge_with_hard_deletes_tagged_for_cognee():
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    source = evernote_source(resource_name="evernote_notes", service=FakeEvernote())

    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "evernote"
    assert getattr(source, PIPELINE_SCOPE_ATTR) == "evernote_notes"
    assert source.write_disposition == "merge"
    assert source.columns["_deleted"]["hard_delete"] is True


def test_a_real_dlt_pipeline_keeps_the_progress_across_runs(tmp_path):
    """State lives in dlt: a rate-limited first run's pass-1 progress survives to the next run."""
    import dlt

    fake = account(CHUNK_SIZE + 5)
    fake.fail("get_filtered_sync_chunk", _limited(), call=2)
    pipeline = dlt.pipeline(
        pipeline_name="evernote_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'e.db'}"),
        dataset_name="evernote",
        pipelines_dir=str(tmp_path / "pipelines"),
    )

    pipeline.run(evernote_source(resource_name="evernote_notes", service=fake))
    fake.calls.clear()
    second = evernote_source(resource_name="evernote_notes", service=fake)
    pipeline.run(second)

    assert fake.calls_to("get_filtered_sync_chunk")[0][0] == CHUNK_SIZE
    assert second.cognee_sync_stats["emitted"] == CHUNK_SIZE + 5


# -- review findings -----------------------------------------------------------------------------


def _down():
    return EvernoteUnavailableError("Evernote did not answer after 4 attempts")


def test_an_outage_on_a_body_cuts_the_run_and_moves_only_the_two_failed_notes_to_retry():
    """F7: every pending note used to go on the retry list, about 7 s of retries each, and the
    sync moved on as if it had finished. Now two failures in a row cut the run; those two notes
    wait in retry (F29) and the rest stay pending."""
    fake = account(5)
    fake.fail("get_note", _down(), call=2, times=10**6)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert _ids(rows) == ["n1"] and len(fake.calls_to("get_note")) == 3, "two in a row: cut"
    assert stats == {**new_stats(), "emitted": 1, "failed": 3, "failed_unavailable": 1}
    assert sorted(state["retry"]) == ["n2", "n3"] and "last_usn" not in state
    assert [note["guid"] for note in state["listing"]["pending"]] == ["n4", "n5"]
    fake._failures.clear()
    rows, _ = _sync(fake, state)
    assert _ids(rows) == ["n4", "n5", "n2", "n3"] and "retry" not in state


def test_an_outage_in_the_listing_cuts_and_keeps_the_progress():
    fake = account(CHUNK_SIZE + 5)
    fake.fail("get_filtered_sync_chunk", _down(), call=2)
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert rows == [] and stats["failed_unavailable"] == 1
    assert state["listing"]["usn"] == CHUNK_SIZE


def test_switching_back_after_a_cut_selection_change_lists_the_whole_account():
    """F8: the cut pass for Home had already forgotten the Work notes, and the run back to Work
    was incremental, so they never came back and Home's emitted note stayed."""
    fake = account(3)
    fake.add_note("h1", "Home one", notebook="nb2")
    fake.add_note("h2", "Home two", notebook="nb2")
    state: dict = {}
    _sync(fake, state, notebook_names=["Work"])
    rows, _ = _sync(fake, state, notebook_names=["Home"], max_notes_per_run=1)
    assert sorted(_ids(rows, deleted=True)) == ["n1", "n2", "n3"] and _ids(rows) == ["h1"]

    rows, _ = _sync(fake, state, notebook_names=["Work"])

    assert _ids(rows) == ["n1", "n2", "n3"] and _ids(rows, deleted=True) == ["h1"]
    assert set(state["note_ids"]) == {"n1", "n2", "n3"}
    fake.add_note("n9", "new in Work")
    rows, _ = _sync(fake, state, notebook_names=["Work"])
    assert _ids(rows) == ["n9"]


def test_a_note_moved_into_a_notebook_that_is_then_expunged_is_deleted():
    """F9: the note was recorded under its old notebook, so the expunge sweep missed it."""
    fake = account(2)
    state: dict = {}
    _sync(fake, state)
    fake.add_notebook("nb3", "Scratch")
    fake.edit_note("n1", notebook="nb3")
    fake.expunge_notebook("nb3")

    rows, _ = _sync(fake, state)

    assert _ids(rows, deleted=True) == ["n1"]
    assert set(state["note_ids"]) == {"n2"}


def test_a_permission_denial_on_the_listing_is_not_reported_as_a_rejected_token():
    """F10: it raised as a rejected token and counted in failed_auth."""
    fake = account(2)
    fake.fail("get_filtered_sync_chunk", EvernotePermissionError("Evernote denied access"))
    stats = new_stats()

    with pytest.raises(EvernotePermissionError):
        list(_iter_rows(fake, {}, stats))
    assert stats == {**new_stats(), "failed": 1}


@pytest.mark.parametrize("name", ["Work Notes", "workNotes", "Work-Notes"])
def test_a_resource_name_dlt_would_rename_is_refused(name):
    """F11: dlt stores these as one table, work_notes, while state and bind use the raw name."""
    with pytest.raises(ValueError, match="must be a dlt table name"):
        evernote_source(resource_name=name, service=FakeEvernote())
    evernote_source(resource_name="work_notes", service=FakeEvernote())


def test_each_capped_run_emits_its_budget_and_a_failed_note_counts_against_it():
    """F14."""
    fake = account(10)
    state: dict = {}
    per_run = [_sync(fake, state, max_notes_per_run=4)[1]["emitted"] for _ in range(3)]
    assert per_run == [4, 4, 2]

    fake = account(3)
    fake.fail("get_note", _broken(), guid="n1")
    rows, stats = _sync(fake, {}, max_notes_per_run=2)
    assert _ids(rows) == ["n2"] and stats["emitted"] == 1 and stats["failed"] == 1


def test_a_tag_selection_takes_a_note_with_any_of_the_tags():
    """F15."""
    fake = account(0)
    fake.add_note("a", "urgent only", tags=["t1"])
    fake.add_note("b", "later only", tags=["t2"])
    fake.add_note("c", "untagged")

    rows, _ = _sync(fake, {}, tag_names=["urgent", "later"])

    assert _ids(rows) == ["a", "b"]


def test_an_expunged_selected_tag_fails_the_run_by_name():
    """F16."""
    fake = account(0)
    fake.add_note("a", "tagged", tags=["t1"])
    state: dict = {}
    _sync(fake, state, tag_names=["urgent"])
    fake.expunge_tag("t1")

    with pytest.raises(EvernoteError, match=r"Tag name\(s\) not found.*\['urgent'\]"):
        _sync(fake, state, tag_names=["urgent"])


def test_a_cut_sync_keeps_no_note_text_in_state():
    """F18: titles of every pending note were committed with each cut run."""
    fake = account(5)
    state: dict = {}

    _sync(fake, state, max_notes_per_run=1)

    assert state["listing"]["pending"][0] == {"guid": "n2", "usn": 6, "notebook": "nb1"}
    assert "Note n" not in json.dumps(state)


def test_a_listed_note_without_a_usable_guid_is_skipped():
    """F19."""
    fake = account(1)
    original = fake.get_filtered_sync_chunk

    def with_broken_note(after_usn, max_entries, check_active=None):
        chunk = original(after_usn, max_entries, check_active)
        if chunk.notes:
            chunk.notes.append(SimpleNamespace(guid=None, updateSequenceNum=1, active=True))
            chunk.notes.append(SimpleNamespace(guid="", updateSequenceNum=1, active=True))
        return chunk

    fake.get_filtered_sync_chunk = with_broken_note

    rows, stats = _sync(fake, {})

    assert _ids(rows) == ["n1"] and stats["failed"] == 0


def test_a_service_for_the_other_environment_is_refused():
    """F21: the bind names the environment, so it has to be the client's."""
    service = FakeEvernote()
    service.sandbox = True
    with pytest.raises(ValueError, match="sandbox differs"):
        evernote_source(resource_name="e", service=service)
    evernote_source(resource_name="e", service=service, sandbox=True)


@pytest.mark.parametrize(
    "error",
    [
        EvernoteError("Evernote user error TAKEN_DOWN (20)"),
        EvernoteError("Evernote user error BAD_DATA_FORMAT (2)"),
        EvernoteError("Evernote API call failed: TProtocolException"),
        EvernotePermissionError("Evernote denied access (PERMISSION_DENIED (3))"),
        EvernoteUnavailableError("Evernote system error INTERNAL_ERROR (4)"),
    ],
    ids=["taken_down", "bad_data", "decode", "permission", "system_error"],
)
def test_one_note_that_always_fails_never_holds_the_others(error):
    """F23: a user error on one body raised every run, and a system error on it cut every run
    at the same first note, so nothing behind it ever synced."""
    fake = account(4)
    fake.fail("get_note", error, times=10**6, guid="n1")
    state: dict = {}
    emitted: list[str] = []

    completed = []
    for _ in range(3):
        rows, _ = _sync(fake, state)
        emitted += _ids(rows)
        completed.append("listing" not in state)

    assert sorted(emitted) == ["n2", "n3", "n4"]
    assert completed[1], "by the second run the note is on retry and the sync completes"
    assert "n1" in state.get("retry", {}) or [
        note["guid"] for note in state["listing"]["pending"]
    ] == ["n1"]


def test_a_cut_listing_keeps_no_note_text_in_state():
    """F24: the saved pass 1 holds selection data only."""
    fake = account(CHUNK_SIZE + 5)
    fake.fail("get_filtered_sync_chunk", _limited(), call=2)
    state: dict = {}

    _sync(fake, state)

    assert state["listing"]["notes"] and "Note n" not in json.dumps(state)


def test_a_note_moved_between_the_listing_and_its_fetch_is_recorded_where_it_is_now():
    """F25: the row and note_ids take the notebook getNote reports, not the listing's."""
    fake = account(1)
    original = fake.get_note

    def moved_meanwhile(guid, check_active=None):
        fake.notes[guid]["notebook"] = "nb2"
        return original(guid, check_active)

    fake.get_note = moved_meanwhile
    state: dict = {}

    rows, _ = _sync(fake, state)

    assert rows[0]["content"].startswith("Notebook: Home\n")
    assert state["note_ids"] == {"n1": "nb2"}


def test_a_note_that_always_has_a_system_error_never_cuts_a_run():
    """F26: the "failed before" mark lived in the saved pass and went with it, so every second
    run was cut by the same note. Every run completes now. Once the other notes are synced, a
    run whose only read is this note cannot tell it from an outage and reports one (F30)."""
    fake = account(4)
    fake.fail("get_note", _down(), times=10**6, guid="n1")
    state: dict = {}
    emitted: list[str] = []

    for run in range(5):
        rows, stats = _sync(fake, state)
        emitted += _ids(rows)
        assert "listing" not in state, "the run completed"
        assert stats["failed_unavailable"] == (run > 0) and stats["failed"] == 1

    assert sorted(emitted) == ["n2", "n3", "n4"] and list(state["retry"]) == ["n1"]


def test_an_outage_over_several_runs_stays_an_outage_and_recovers_whole():
    """F27: a long outage ended as one failed note per run instead of an outage. Every outage
    run now reports failed_unavailable; at most two notes a run move to retry (option A, so
    that two notes that always fail cannot hold the pass), and nothing is lost."""
    fake = account(5)
    fake.fail("get_note", _down(), times=10**6)
    state: dict = {}

    waiting = 0
    for _ in range(8):
        rows, stats = _sync(fake, state)
        assert rows == [] and stats["failed_unavailable"] == 1
        now = len(state.get("retry", {}))
        assert now - waiting <= 2
        waiting = now
    assert waiting == 5 and "listing" not in state

    fake._failures.clear()
    rows, _ = _sync(fake, state)
    assert sorted(_ids(rows)) == ["n1", "n2", "n3", "n4", "n5"] and "listing" not in state


def test_a_note_held_after_an_unavailable_answer_is_kept_pending_when_the_run_is_cut():
    """The held note is no failure yet: a cut before the next read puts it back in front."""
    fake = account(3)
    fake.fail("get_note", _down(), guid="n1")
    fake.fail("get_note", _limited(), guid="n2")
    state: dict = {}

    rows, stats = _sync(fake, state)

    assert rows == [] and stats["failed_rate_limit"] == 1 and "retry" not in state
    assert [note["guid"] for note in state["listing"]["pending"]] == ["n1", "n2", "n3"]


def test_a_note_failing_with_a_system_error_counts_against_the_budget():
    """F28: each attempt is a request, so it uses the run's max_notes_per_run."""
    fake = account(3)
    fake.fail("get_note", _down(), guid="n1")

    rows, _ = _sync(fake, {}, max_notes_per_run=2)

    assert _ids(rows) == ["n2"]


# -- the retry tail (F29, F30) -------------------------------------------------------------------


def test_two_notes_that_always_fail_never_hold_new_notes_and_edits():
    """F29: both bad notes failed back to back, the run was cut as an outage, and the pass was
    never finished, so new notes and edits were never read again. The cut now puts both in
    retry; the pass completes on the next run and the retry tail keeps trying them."""
    fake = account(4)
    for guid in ("n1", "n2"):
        fake.fail("get_note", _down(), times=10**6, guid=guid)
    state: dict = {}
    _sync(fake, state)
    fake.add_note("n9", "new")
    fake.edit_note("n3", "edited")

    seen: set[str] = set()
    for _ in range(4):
        rows, _ = _sync(fake, state)
        seen |= set(_ids(rows))
        assert "listing" not in state, "every run completes its pass"

    assert state["last_usn"] == fake.update_count
    assert seen == {"n3", "n4", "n9"}, "n4 left from the cut first sync, the edit, the new note"
    assert sorted(state["retry"]) == ["n1", "n2"]


def test_an_outage_with_one_note_waiting_for_a_retry_keeps_it_and_the_cursor():
    """F30: Evernote down with one note to read is reported as an outage; the note stays in
    retry, the cursor is fine, and the note is read once Evernote answers."""
    fake = account(2)
    fake.fail("get_note", _broken(), guid="n1")
    state: dict = {}
    _sync(fake, state)
    usn = state["last_usn"]
    fake.fail("get_note", _down(), times=10**6)

    rows, stats = _sync(fake, state)

    assert rows == [] and stats["failed"] == 1 and stats["failed_unavailable"] == 1
    assert list(state["retry"]) == ["n1"] and state["last_usn"] == usn
    fake._failures.clear()
    rows, _ = _sync(fake, state)
    assert _ids(rows) == ["n1"] and "retry" not in state


def test_an_outage_in_the_retry_tail_ends_the_tail_and_keeps_the_pass():
    fake = account(4)
    for guid in ("n1", "n2", "n3"):
        fake.fail("get_note", _broken(), guid=guid)
    state: dict = {}
    _sync(fake, state)
    fake.add_note("n9", "new")
    fake.fail("get_note", _down(), times=10**6, guid="n1")
    fake.fail("get_note", _down(), times=10**6, guid="n2")

    rows, stats = _sync(fake, state)

    assert _ids(rows) == ["n9"] and "listing" not in state
    assert stats["failed_unavailable"] == 1 and stats["failed"] == 2
    assert sorted(state["retry"]) == ["n1", "n2", "n3"], "n3 waits: the tail ended at the outage"


def test_a_rate_limit_or_a_rejected_token_in_the_retry_tail_is_a_clean_cut():
    for error, key in ((_limited(), "failed_rate_limit"), (_rejected(), "failed_auth")):
        fake = account(2)
        fake.fail("get_note", _broken(), guid="n1")
        state: dict = {}
        _sync(fake, state)
        fake.fail("get_note", error, guid="n1")

        rows, stats = _sync(fake, state)  # idle pass: nothing new, then the tail

        assert rows == [] and stats[key] == 1 and stats["failed"] == 1
        assert list(state["retry"]) == ["n1"]


def test_the_retry_tail_reads_notes_within_the_runs_budget():
    fake = account(3)
    for guid in ("n1", "n2"):
        fake.fail("get_note", _broken(), guid=guid)
    state: dict = {}
    _sync(fake, state)
    fake.add_note("n9", "new")
    fake.calls.clear()

    rows, _ = _sync(fake, state, max_notes_per_run=2)

    assert _ids(rows) == ["n9", "n1"] and list(state["retry"]) == ["n2"]
    assert len(fake.calls_to("get_note")) == 2


def test_notes_that_always_fail_cannot_starve_the_retry_tail_under_a_budget():
    """F31: the tail read retry in guid order, so with a cap the failing notes sorting first
    used the budget every run and the healthy note behind them was never read again."""
    fake = account(3)
    for guid in ("n1", "n2"):
        fake.fail("get_note", _broken(), times=10**6, guid=guid)
    fake.fail("get_note", _broken(), guid="n3")
    state: dict = {}
    _sync(fake, state)
    assert list(state["retry"]) == ["n1", "n2", "n3"]

    seen: list[str] = []
    for _ in range(2):
        rows, _ = _sync(fake, state, max_notes_per_run=2)
        seen += _ids(rows)

    assert seen == ["n3"] and sorted(state["retry"]) == ["n1", "n2"]


def test_the_retry_order_survives_dlt_state(tmp_path):
    """The rotation lives in the order of state["retry"]: a dlt round trip must keep it."""
    import dlt

    pipeline = dlt.pipeline(
        pipeline_name="evernote_order",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'o.db'}"),
        dataset_name="evernote",
        pipelines_dir=str(tmp_path / "pipelines"),
    )

    @dlt.resource(name="order_probe")
    def write():
        dlt.current.resource_state()["retry"] = {"z": 1, "a": 2, "m": 3}
        yield {"id": "x"}

    pipeline.run(write())
    seen = []

    @dlt.resource(name="order_probe")
    def read():
        seen.append(list(dlt.current.resource_state()["retry"]))
        yield {"id": "x"}

    pipeline.run(read())
    assert seen == [["z", "a", "m"]]


def test_an_outage_with_one_note_pending_in_the_pass_is_reported_and_recovers():
    """F30: the "two in a row" rule never fired with one note, so an outage read as a note
    failure. The note waits in retry like a pair would; the next run completes the pass."""
    fake = account(1)
    state: dict = {}
    _sync(fake, state)
    fake.edit_note("n1", "edited")
    fake.fail("get_note", _down(), times=10**6)

    rows, stats = _sync(fake, state)

    assert rows == [] and stats["failed_unavailable"] == 1
    assert list(state["retry"]) == ["n1"]
    fake._failures.clear()
    rows, _ = _sync(fake, state)
    assert _ids(rows) == ["n1"] and "listing" not in state and "retry" not in state
    assert state["last_usn"] == fake.update_count


def test_a_capped_run_keeps_one_read_for_a_note_waiting_in_retry():
    """F32: new notes filled max_notes_per_run every run, so the retry tail never ran."""
    fake = account(1)
    fake.fail("get_note", _broken(), guid="n1")
    state: dict = {}
    _sync(fake, state)
    passes: list[int] = []
    retried = []
    for run in range(2):
        fake.add_note(f"a{run}", "new")
        fake.add_note(f"b{run}", "new")
        rows, _ = _sync(fake, state, max_notes_per_run=2)
        ids = _ids(rows)
        passes.append(len([i for i in ids if i != "n1"]))
        retried.append("n1" in ids)

    assert retried[0], "the waiting note is read in the first capped run"
    assert all(count >= 1 for count in passes), "the pass still moves every run"


def test_a_cap_of_one_keeps_its_read_for_the_pass():
    """With room for one read only, reserving it for the tail would stop the pass for good."""
    fake = account(1)
    fake.fail("get_note", _broken(), times=10**6, guid="n1")
    state: dict = {}
    _sync(fake, state)
    fake.add_note("a", "new")
    fake.add_note("b", "new")

    seen = []
    for _ in range(2):
        rows, _ = _sync(fake, state, max_notes_per_run=1)
        seen += _ids(rows)

    assert seen == ["a", "b"]


def test_the_retry_tail_after_a_capped_pass_renders_with_the_listings_names():
    """The tail also runs when the cap cuts the pass, before the pass stores its notebook and
    tag names: it rendered from state, which had none yet, and raised KeyError."""
    fake = account(0)
    fake.add_note("n1", "tagged", tags=["t1"])
    for guid in ("n2", "n3", "n4", "n5"):
        fake.add_note(guid, "plain")
    fake.fail("get_note", _broken(), guid="n1")
    state: dict = {}
    _sync(fake, state, max_notes_per_run=2)  # n1 fails, n2 is read, the cap cuts the pass
    assert "notebooks" not in state and list(state["retry"]) == ["n1"]

    rows, _ = _sync(fake, state, max_notes_per_run=2)

    (row,) = [r for r in rows if r["id"] == "note:n1"]
    assert row["content"].startswith("Notebook: Work\nTags: urgent\n")
