# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK: one folder id per source, the real client over a fake.
"""The sync state machine against an in-memory Dropbox, through the real httpx client.

No network, no LLM. Each test drives the fake through a real-world change and checks the rows
the connector yields and the state it keeps for the next run.
"""

import copy
import logging
import random

import pytest

from cognee.tasks.ingestion.connectors.dropbox import (
    STATS_KEYS,
    DropboxAccountMismatchError,
    DropboxApiError,
    DropboxAuthError,
    DropboxFolderNotFoundError,
    _iter_rows,
    _Session,
)
from cognee.tests.unit.tasks.connectors.fake_dropbox import (
    ACCOUNT,
    FakeDropbox,
    lookup_error,
    make_pdf,
)


@pytest.fixture
def fake():
    return FakeDropbox()


@pytest.fixture
def waits():
    return []


def sync(fake, state, folder_id="", *, max_mb=None, check_active=None, waits=None, account=ACCOUNT):
    stats = dict.fromkeys(STATS_KEYS, 0)
    client = fake.client(sleep=(waits if waits is not None else []).append)
    rows = list(
        _iter_rows(
            _Session(client, check_active),
            folder_id=folder_id,
            account_id=account,
            state=state,
            stats=stats,
            max_bytes=None if max_mb is None else max_mb * 1024 * 1024,
        )
    )
    upserts = {row["id"]: row for row in rows if not row["_deleted"]}
    deletes = sorted(row["id"] for row in rows if row["_deleted"])
    return upserts, deletes, stats


def run_with_stats(fake, state, folder_id="", **kwargs):
    stats = dict.fromkeys(STATS_KEYS, 0)
    with pytest.raises(Exception) as error:
        list(
            _iter_rows(
                _Session(fake.client(sleep=lambda _s: None), kwargs.get("check_active")),
                folder_id=folder_id,
                account_id=ACCOUNT,
                state=state,
                stats=stats,
                max_bytes=None,
            )
        )
    return error.value, stats


# ---------------------------------------------------------------- first sync and rows
def test_first_sync_yields_every_supported_file_and_binds_the_state(fake):
    notes = fake.mkdir("/Notes")
    a = fake.put("/Notes/a.md", "# Alpha notes")
    b = fake.put("/Notes/2026/b.txt", "bravo text")
    fake.put("/Elsewhere/c.md", "not selected")
    state: dict = {}

    upserts, deletes, stats = sync(fake, state, notes)

    assert set(upserts) == {a, b} and deletes == []
    assert state["v"] == 1 and state["account"] == ACCOUNT and state["folder"] == notes
    assert state["index"] == {"/notes/a.md": a, "/notes/2026/b.txt": b}
    assert state["path"] == "/notes" and isinstance(state["cursor"], str)
    assert stats["complete"] == 1 and stats["failed"] == 0 and stats["scanned"] == 2


def test_the_row_is_title_text_and_structure_up_to_the_selected_folder(fake):
    notes = fake.mkdir("/Notes")
    year = fake.mkdir("/Notes/2026")
    a = fake.put("/Notes/2026/plan.md", "Ship the connector.")

    upserts, _, _ = sync(fake, {}, notes)

    assert upserts[a] == {
        "id": a,
        "title": "plan.md",
        "content": "Modified: 2026-01-01\n\nShip the connector.",
        "url": None,
        "cognee_structure": {
            "ancestors": [
                {"kind": "folder", "id": year, "name": "2026"},
                {"kind": "folder", "id": notes, "name": "Notes"},
            ]
        },
        "_deleted": False,
    }


def test_a_file_at_the_top_of_the_whole_dropbox_has_no_ancestors(fake):
    a = fake.put("/a.md", "alpha")

    upserts, _, _ = sync(fake, {}, "")

    assert upserts[a]["cognee_structure"] == {"ancestors": []}


def test_every_stats_key_exists_on_every_exit(fake):
    fake.put("/a.md", "alpha")
    _, _, stats = sync(fake, {})
    assert set(stats) == set(STATS_KEYS)

    fake.queued["files/list_folder"] = [(429, {}, {"Retry-After": "3600"})]
    _, _, stats = sync(fake, {})
    assert set(stats) == set(STATS_KEYS) and stats["failed_rate_limit"] == 1


def test_an_unchanged_resync_yields_nothing_and_saves_a_fresh_cursor(fake):
    fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)
    first = state["cursor"]
    fake.put("/photo.png", b"\x89PNG")

    upserts, deletes, stats = sync(fake, state)

    assert upserts == {} and deletes == []
    assert state["cursor"] != first and stats["unchanged"] == 1


def test_pagination_reads_every_page(fake):
    fake.page_size = 2
    ids = {fake.put(f"/f{i}.txt", f"file {i}") for i in range(5)}
    state: dict = {}
    assert set(sync(fake, state)[0]) == ids

    more = {fake.put(f"/g{i}.txt", f"file {i}") for i in range(5)}
    assert set(sync(fake, state)[0]) == more


def test_a_shuffled_listing_gives_the_same_rows(fake, monkeypatch):
    for i in range(8):
        fake.put(f"/Dir{i % 3}/f{i}.md", f"file {i}")
    expected, _, _ = sync(fake, {})

    original = fake._page

    def shuffled(entries, final):
        entries = list(entries)
        folders = [e for e in entries if e[".tag"] == "folder"]
        rest = [e for e in entries if e[".tag"] != "folder"]
        random.Random(7).shuffle(rest)
        return original(folders + rest, final)

    monkeypatch.setattr(fake, "_page", shuffled)
    again, _, _ = sync(fake, {})

    assert list(again.values()) == list(expected.values())


# ---------------------------------------------------------------- edits, moves, deletes
def test_an_edit_reyields_the_same_id_with_new_content(fake):
    a = fake.put("/a.md", "old")
    state: dict = {}
    sync(fake, state)

    assert fake.put("/a.md", "new") == a
    upserts, deletes, _ = sync(fake, state)

    assert list(upserts) == [a] and upserts[a]["content"].endswith("new") and deletes == []


def test_a_rename_inside_the_folder_is_one_row_with_the_new_path(fake):
    notes = fake.mkdir("/Notes")
    a = fake.put("/Notes/Inbox/a.md", "alpha")
    state: dict = {}
    sync(fake, state, notes)

    fake.move("/Notes/Inbox/a.md", "/Notes/Archive/renamed.md")
    upserts, deletes, _ = sync(fake, state, notes)

    assert deletes == []
    assert upserts[a]["title"] == "renamed.md"
    assert upserts[a]["content"] == "Modified: 2026-01-01\n\nalpha"
    assert upserts[a]["cognee_structure"]["ancestors"][0]["name"] == "Archive"
    assert state["index"] == {"/notes/archive/renamed.md": a}


def test_a_case_only_rename_keeps_the_id_and_shows_the_new_case(fake):
    a = fake.put("/plan.md", "alpha")
    state: dict = {}
    sync(fake, state)

    fake.move("/plan.md", "/Plan.md")
    upserts, deletes, _ = sync(fake, state)

    assert deletes == [] and upserts[a]["title"] == "Plan.md"
    assert upserts[a]["content"] == "Modified: 2026-01-01\n\nalpha"
    assert state["index"] == {"/plan.md": a}


def test_a_cross_folder_move_tombstones_in_one_source_and_adds_in_the_other(fake):
    notes, work = fake.mkdir("/Notes"), fake.mkdir("/Work")
    a = fake.put("/Notes/a.md", "alpha")
    notes_state: dict = {}
    work_state: dict = {}
    sync(fake, notes_state, notes)
    sync(fake, work_state, work)

    fake.move("/Notes/a.md", "/Work/a.md")
    _, notes_deletes, notes_stats = sync(fake, notes_state, notes)
    work_upserts, work_deletes, _ = sync(fake, work_state, work)

    assert notes_deletes == [a] and notes_stats["deleted"] == 1
    assert list(work_upserts) == [a] and work_deletes == []


def test_moving_a_file_out_of_the_folder_forgets_it(fake):
    synced = fake.mkdir("/Synced")
    a = fake.put("/Synced/a.md", "alpha")
    state: dict = {}
    sync(fake, state, synced)

    fake.move("/Synced/a.md", "/Elsewhere/a.md")
    upserts, deletes, _ = sync(fake, state, synced)

    assert upserts == {} and deletes == [a]


def test_renaming_the_selected_folder_keeps_the_state_and_reyields_the_new_paths(fake):
    notes = fake.mkdir("/Notes")
    a = fake.put("/Notes/a.md", "alpha")
    state: dict = {}
    sync(fake, state, notes)

    fake.move("/Notes", "/Journal")
    fake.calls.clear()
    upserts, deletes, stats = sync(fake, state, notes)

    # How Dropbox replays a cursor taken under the old path is not documented, so a changed
    # path is listed whole and reconciled like a snapshot.
    assert "files/list_folder" in fake.calls

    assert deletes == [] and list(upserts) == [a]
    assert upserts[a]["content"] == "Modified: 2026-01-01\n\nalpha", "only the structure changed"
    assert upserts[a]["cognee_structure"]["ancestors"] == [
        {"kind": "folder", "id": notes, "name": "Journal"}
    ]
    assert state["path"] == "/journal" and state["folder"] == notes
    assert stats["failed"] == 0


def test_a_deleted_file_is_tombstoned_by_id_from_the_index(fake):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    state: dict = {}
    sync(fake, state)

    fake.delete("/b.md")
    upserts, deletes, stats = sync(fake, state)

    assert upserts == {} and deletes == [b] and stats["deleted"] == 1
    assert state["index"] == {"/a.md": a}


def test_a_deleted_subfolder_forgets_everything_under_it_but_not_lookalike_siblings(fake):
    a = fake.put("/Notes/a.md", "alpha")
    b = fake.put("/Notes/deep/b.md", "bravo")
    keep = fake.put("/Notes2/keep.md", "keep")
    state: dict = {}
    sync(fake, state)

    fake.delete("/Notes")
    _, deletes, _ = sync(fake, state)

    assert deletes == sorted([a, b]) and state["index"] == {"/notes2/keep.md": keep}


def test_a_file_added_and_deleted_in_one_round_is_never_downloaded(fake):
    state: dict = {}
    sync(fake, state)
    temp = fake.put("/temp.md", "short lived")
    fake.delete("/temp.md")

    upserts, deletes, _ = sync(fake, state)

    assert upserts == {} and fake.downloads == [] and deletes == [temp]


def test_a_path_reused_by_a_new_file_forgets_the_old_id(fake):
    old = fake.put("/a.md", "old file")
    state: dict = {}
    sync(fake, state)

    fake.vanish("/a.md")
    new = fake.put("/a.md", "brand new file")
    upserts, deletes, _ = sync(fake, state)

    assert deletes == [old] and list(upserts) == [new]


def test_a_folder_created_where_a_file_was_forgets_the_file(fake):
    old = fake.put("/plans", "a file called plans")
    state: dict = {}
    sync(fake, state)

    fake.vanish("/plans")
    inner = fake.put("/Plans/q1.md", "now a folder")
    upserts, deletes, _ = sync(fake, state)

    assert deletes == [old] and list(upserts) == [inner]


def test_a_reset_cursor_relists_and_reconciles_like_a_snapshot(fake):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    state: dict = {}
    sync(fake, state)

    fake.delete("/b.md")
    c = fake.put("/c.md", "charlie")
    fake.reset_cursors.add(state["cursor"])
    upserts, deletes, stats = sync(fake, state)

    # The unchanged file is relisted but not emitted again: its bytes and path are the same.
    assert deletes == [b] and set(upserts) == {c} and stats["unchanged"] == 1
    assert a in state["index"].values()


def test_a_change_without_a_path_is_skipped_not_fatal(fake):
    a = fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)
    fake.log.append({".tag": "deleted", "name": "unmounted"})

    upserts, deletes, _ = sync(fake, state)

    assert upserts == {} and deletes == [] and state["index"] == {"/a.md": a}


# ---------------------------------------------------------------- the selected folder
def test_a_folder_id_never_listed_that_does_not_exist_raises(fake):
    state: dict = {}

    error, stats = run_with_stats(fake, state, "id:typo")

    assert isinstance(error, DropboxFolderNotFoundError)
    assert state == {} and stats["failed"] == 1


def test_a_synced_folder_that_is_gone_is_forgotten_like_a_delete(fake):
    project = fake.mkdir("/Project")
    a = fake.put("/Project/a.md", "alpha")
    state: dict = {}
    sync(fake, state, project)

    fake.delete("/Project")
    upserts, deletes, stats = sync(fake, state, project)

    assert upserts == {} and deletes == [a] and stats["deleted"] == 1
    assert "cursor" not in state and state["index"] == {} and state["gone"] is True
    # Later runs know it went away: counted on their own key, never taken for a typo.
    for _ in range(2):
        upserts, deletes, stats = sync(fake, state, project)
        assert (upserts, deletes) == ({}, [])
        assert (stats["failed"], stats["failed_folder_gone"], stats["failed_listing"]) == (1, 1, 0)


def test_a_gone_folder_that_comes_back_syncs_again_from_scratch(fake):
    project = fake.mkdir("/Project")
    folder_meta = fake.folders["/project"]
    fake.put("/Project/a.md", "alpha")
    state: dict = {}
    sync(fake, state, project)
    fake.delete("/Project")
    sync(fake, state, project)
    sync(fake, state, project)

    fake.folders["/project"] = folder_meta  # the same id resolves again (restored)
    b = fake.put("/Project/b.md", "bravo")
    upserts, _, stats = sync(fake, state, project)

    assert b in upserts and stats["failed"] == 0 and "gone" not in state


def test_a_folder_that_cannot_be_read_raises_and_forgets_nothing(fake):
    project = fake.mkdir("/Project")
    fake.put("/Project/a.md", "alpha")
    state: dict = {}
    sync(fake, state, project)
    before = copy.deepcopy(state)

    fake.queued["files/get_metadata"] = [lookup_error("restricted_content")]
    with pytest.raises(DropboxApiError):
        sync(fake, state, project)

    assert state == before


@pytest.mark.parametrize(
    "answer", [(500, {}), (403, {}), lookup_error("other", "other"), (200, {"no": "entries"})]
)
def test_a_listing_error_raises_and_never_deletes(fake, answer):
    fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)
    before = copy.deepcopy(state)
    fake.delete("/a.md")
    fake.queued["files/list_folder/continue"] = [answer] * 4

    with pytest.raises(DropboxApiError):
        sync(fake, state)

    assert state == before


def test_a_repeated_cursor_raises_instead_of_looping(fake):
    fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)
    stuck = {"entries": [], "cursor": state["cursor"], "has_more": True}
    fake.queued["files/list_folder/continue"] = [(200, stuck)] * 3

    with pytest.raises(DropboxApiError, match="same cursor"):
        sync(fake, state)


def test_another_account_on_the_same_state_is_refused_before_any_request(fake):
    fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)
    fake.calls.clear()

    with pytest.raises(ValueError, match="another Dropbox account"):
        sync(fake, state, account="dbid:someone-else")
    assert fake.calls == []


def test_another_folder_on_the_same_state_is_refused(fake):
    fake.put("/a.md", "alpha")
    other = fake.mkdir("/Other")
    state: dict = {}
    sync(fake, state)

    with pytest.raises(ValueError, match="another Dropbox folder"):
        sync(fake, state, other)


# ---------------------------------------------------------------- failures and give-up (D5)
def test_a_failed_download_waits_by_id_and_is_retried_next_run(fake, waits):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    fake.download_errors[b] = (500, {})
    state: dict = {}

    upserts, _, stats = sync(fake, state, waits=waits)

    assert set(upserts) == {a} and stats["failed"] == 1 and stats["complete"] == 1
    assert stats["failed_retry"] == 1 and state["retry"] == [b]

    del fake.download_errors[b]
    fake.downloads.clear()
    upserts, _, stats = sync(fake, state)

    assert set(upserts) == {b} and fake.downloads == [b] and state["retry"] == []


def test_a_file_that_keeps_failing_is_retried_every_run_and_keeps_its_stored_text(fake):
    a = fake.put("/a.md", "v1")
    state: dict = {}
    sync(fake, state)

    fake.put("/a.md", "v2")
    fake.download_errors[a] = lookup_error("locked")
    results = [sync(fake, state) for _ in range(5)]

    assert [r[1] for r in results] == [[]] * 5, "a failure never deletes memory"
    assert [(r[2]["failed"], r[2]["failed_retry"]) for r in results] == [(1, 1)] * 5
    assert state["retry"] == [a] and a in state["stored"]
    assert fake.downloads.count(a) == 1 + 5, "never given up"


def test_a_new_file_that_fails_five_runs_then_succeeds_is_ingested(fake):
    state: dict = {}
    sync(fake, state)
    a = fake.put("/a.md", "late but here")
    fake.download_errors[a] = (503, {})
    for _ in range(5):
        upserts, _, stats = sync(fake, state)
        assert upserts == {} and stats["failed"] > 0, "the connection stays degraded"

    del fake.download_errors[a]
    upserts, _, stats = sync(fake, state)

    assert upserts[a]["content"].endswith("\n\nlate but here")
    assert stats["failed"] == 0 and state["retry"] == []


def test_the_run_after_an_outage_is_not_clean_while_a_file_is_pending(fake):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    state: dict = {}
    fake.download_errors[b] = (503, {})
    sync(fake, state)

    fake.put("/a.md", "alpha, edited")  # the next run has fresh work as well
    _, _, stats = sync(fake, state)

    assert stats["failed"] == 1 and stats["failed_retry"] == 1 and state["retry"] == [b]
    assert a in state["stored"]


def test_a_retried_file_deleted_in_the_meantime_is_dropped(fake):
    a = fake.put("/a.md", "alpha")
    fake.download_errors[a] = lookup_error("locked")
    state: dict = {}
    sync(fake, state)

    fake.delete("/a.md")
    upserts, deletes, _ = sync(fake, state)

    assert upserts == {} and deletes == [a] and state["retry"] == []


def test_a_retried_file_that_vanished_without_a_delete_is_dropped(fake):
    a = fake.put("/a.md", "alpha")
    fake.download_errors[a] = lookup_error("locked")
    state: dict = {}
    sync(fake, state)
    fake.vanish("/a.md")

    upserts, deletes, stats = sync(fake, state)

    assert upserts == {} and deletes == [] and stats["failed"] == 0 and state["retry"] == []


def test_a_file_gone_between_listing_and_download_is_skipped_not_found(fake):
    a = fake.put("/a.md", "alpha")
    fake.download_errors[a] = lookup_error("not_found")

    _, _, stats = sync(fake, {})

    assert stats["skipped_not_found"] == 1 and stats["failed"] == 0


# ---------------------------------------------------------------- auth and rate limits
def test_a_401_after_rows_keeps_them_cuts_and_the_next_run_resumes(fake):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    c = fake.put("/c.md", "charlie")
    fake.download_errors[b] = (401, {"error": {".tag": "expired_access_token"}})
    state: dict = {}

    upserts, _, stats = sync(fake, state)

    assert set(upserts) == {a}
    assert fake.downloads == [a, b], "no request after the 401"
    assert (stats["failed"], stats["failed_auth"], stats["complete"]) == (1, 1, 0)
    assert state["retry"] == [b, c]

    del fake.download_errors[b]
    upserts, _, stats = sync(fake, state)
    assert set(upserts) == {b, c} and stats["complete"] == 1


def test_a_401_before_anything_was_stored_raises_and_is_counted(fake):
    a = fake.put("/a.md", "alpha")
    fake.download_errors[a] = (401, {"error": {".tag": "invalid_access_token"}})
    state: dict = {}

    error, stats = run_with_stats(fake, state)

    assert isinstance(error, DropboxAuthError)
    assert state == {} and (stats["failed"], stats["failed_auth"]) == (1, 1)


def test_a_401_while_listing_raises_and_is_counted(fake):
    fake.queued["files/list_folder"] = [(401, {"error": {".tag": "invalid_access_token"}})]

    error, stats = run_with_stats(fake, {})

    assert isinstance(error, DropboxAuthError) and stats["failed_auth"] == 1


def test_a_short_retry_after_is_slept_in_steps_and_the_request_retried(fake, waits):
    a = fake.put("/a.md", "alpha")
    fake.queued["files/download"] = [(429, {}, {"Retry-After": "2"})]
    checks: list[int] = []

    upserts, _, stats = sync(fake, {}, waits=waits, check_active=lambda: checks.append(1))

    assert set(upserts) == {a} and waits == [1, 1] and stats["failed_rate_limit"] == 0
    assert fake.calls.count("files/download") == 2


def test_a_long_retry_after_cuts_cleanly_and_forgets_nothing(fake, waits):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    state: dict = {}
    sync(fake, state)
    fake.delete("/a.md")
    fake.put("/b.md", "bravo, edited")
    fake.queued["files/download"] = [(429, {}, {"Retry-After": "3600"})]

    _, deletes, stats = sync(fake, state, waits=waits)

    assert waits == [] and deletes == [a]
    assert (stats["failed"], stats["failed_rate_limit"], stats["complete"]) == (1, 1, 0)
    assert state["retry"] == [b] and b in state["stored"]


def test_a_rate_limit_while_listing_leaves_the_state_as_it_was(fake, waits):
    fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)
    before = copy.deepcopy(state)
    fake.put("/b.md", "beta")
    fake.queued["files/list_folder/continue"] = [(429, {}, {"Retry-After": "3600"})]

    upserts, deletes, stats = sync(fake, state, waits=waits)

    assert upserts == {} and deletes == [] and stats["failed_rate_limit"] == 1
    assert state == before


def test_waits_share_one_budget_per_run(fake, waits):
    ids = [fake.put(f"/f{i:02}.md", f"file {i}") for i in range(5)]
    fake.queued["files/download"] = [(429, {}, {"Retry-After": "60"})] * 10

    upserts, _, stats = sync(fake, {}, waits=waits)

    assert sum(waits) == 120 and max(waits) == 1
    assert upserts == {} or set(upserts) <= set(ids)
    assert stats["failed_rate_limit"] == 1


def test_a_5xx_is_retried_with_1_2_4_seconds_then_fails_the_file(fake, waits):
    a = fake.put("/a.md", "alpha")
    fake.download_errors[a] = (503, {})
    state: dict = {}

    _, _, stats = sync(fake, state, waits=waits)

    assert sum(waits) == 1 + 2 + 4 and fake.downloads == [a] * 4
    assert stats["failed"] == 1 and state["retry"] == [a]


def test_a_4xx_is_not_retried(fake, waits):
    a = fake.put("/a.md", "alpha")
    fake.download_errors[a] = (400, b"bad input")

    sync(fake, {}, waits=waits)

    assert waits == [] and fake.downloads == [a]


def test_a_long_wait_is_cancelled_by_check_active_between_steps(fake, waits):
    fake.put("/a.md", "alpha")
    fake.queued["files/download"] = [(429, {}, {"Retry-After": "30"})]

    def check_active():
        if len(waits) == 3:
            raise PermissionError("revoked")

    with pytest.raises(PermissionError):
        sync(fake, {}, waits=waits, check_active=check_active)
    assert sum(waits) == 3


# ---------------------------------------------------------------- check_active
class _Revoked(Exception):
    pass


def test_check_active_runs_before_every_request(fake):
    fake.page_size = 2
    for i in range(5):
        fake.put(f"/f{i}.md", f"file {i}")
    state: dict = {}

    def check_active():
        if len(fake.calls) >= 2:
            raise _Revoked

    with pytest.raises(_Revoked):
        sync(fake, state, check_active=check_active)

    assert len(fake.calls) == 2 and state == {}


@pytest.mark.parametrize("error", [PermissionError, TimeoutError, ConnectionError])
def test_check_active_errors_stop_the_run_as_they_are(fake, error):
    fake.put("/a.md", "alpha")

    def check_active():
        if fake.calls[-1:] == ["files/list_folder"]:
            raise error("revoked")

    with pytest.raises(error):
        sync(fake, {}, check_active=check_active)
    assert "files/download" not in fake.calls


# ---------------------------------------------------------------- content and skips
def test_pdf_text_is_extracted(fake):
    report = fake.put("/report.pdf", make_pdf("Quarterly report Zephyrcorp"))

    upserts, _, _ = sync(fake, {})

    assert "Quarterly report Zephyrcorp" in upserts[report]["content"]


def test_a_paper_doc_is_exported_as_markdown(fake):
    info = {"export_as": "markdown", "export_options": ["markdown", "html"]}
    paper = fake.put("/Plans.paper", "# Roadmap", downloadable=False, export_info=info)

    upserts, _, _ = sync(fake, {})

    assert fake.exports == [(paper, "markdown")] and "# Roadmap" in upserts[paper]["content"]


def test_skips_are_counted_by_reason(fake):
    good = fake.put("/good.md", "fine")
    fake.put("/photo.png", b"\x89PNG")
    fake.put("/huge.txt", "x", size=2 * 1024 * 1024)
    fake.put("/empty.md", "   \n")
    fake.put("/broken.pdf", b"not a pdf at all")
    fake.put("/binary.csv", b"a,b\x00\x01")
    fake.put("/Sheet.gsheet", "a", downloadable=False, export_info={"export_as": "xlsx"})

    upserts, _, stats = sync(fake, {}, max_mb=1)

    assert set(upserts) == {good}
    assert stats["scanned"] == 7 and stats["skipped"] == 6
    assert stats["skipped_unsupported_type"] == 2 and stats["skipped_too_large"] == 1
    assert stats["skipped_empty_content"] == 1 and stats["skipped_unparseable"] == 2
    assert stats["failed"] == 0


def test_no_size_cap_by_default(fake):
    big = fake.put("/big.txt", "x" * (3 * 1024 * 1024))

    upserts, _, stats = sync(fake, {})

    assert big in upserts and stats["skipped_too_large"] == 0


def test_an_exported_paper_doc_is_held_to_the_size_limit_and_never_downloaded(fake):
    info = {"export_as": "markdown"}
    fake.put("/Big.paper", "# Plan", downloadable=False, export_info=info, size=3 * 1024 * 1024)

    _, _, stats = sync(fake, {}, max_mb=1)

    assert stats["skipped_too_large"] == 1 and fake.exports == []


@pytest.mark.parametrize(
    "change",
    [
        lambda fake: fake.put("/a.md", "   \n  "),
        lambda fake: fake.put("/a.md", "x" * 2 * 1024 * 1024),
    ],
)
def test_a_stored_file_that_is_now_skipped_is_forgotten(fake, change):
    a = fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state, max_mb=1)

    change(fake)
    upserts, deletes, _ = sync(fake, state, max_mb=1)

    assert upserts == {} and deletes == [a]


def test_a_new_file_that_is_skipped_emits_no_tombstone(fake):
    fake.put("/photo.png", b"\x89PNG")

    _, deletes, stats = sync(fake, {})

    assert deletes == [] and stats["deleted"] == 0


def test_utf16_text_is_decoded(fake):
    notepad = fake.put("/notepad.txt", "hello from notepad".encode("utf-16"))

    upserts, _, _ = sync(fake, {})

    assert "hello from notepad" in upserts[notepad]["content"]


def test_an_unsupported_file_answer_is_skipped_not_retried(fake):
    a = fake.put("/locked.md", "no")
    fake.download_errors[a] = lookup_error("unsupported_file", None)
    state: dict = {}

    _, _, stats = sync(fake, state)

    assert stats["skipped_unsupported_type"] == 1 and state["retry"] == []


# ---------------------------------------------------------------- privacy
def test_no_names_paths_ids_or_token_reach_the_logs(fake, caplog):
    caplog.set_level(logging.DEBUG)
    secret = fake.mkdir("/Secret Plans")
    budget = fake.put("/Secret Plans/budget-2027.md", "numbers")
    offline = fake.put("/Secret Plans/offline.md", "later")
    fake.download_errors[offline] = (503, {})
    state: dict = {}

    sync(fake, state, secret)
    fake.reset_cursors.add(state["cursor"])
    sync(fake, state, secret)
    fake.delete("/Secret Plans")
    sync(fake, state, secret)

    assert caplog.text
    for value in ("Secret Plans", "secret plans", "budget-2027", "offline.md", "sl.fake-token"):
        assert value not in caplog.text
    for value in (budget, offline, secret):
        assert value not in caplog.text


def test_a_folder_moved_between_the_lookup_and_the_listing_stops_without_changing_state(
    fake, monkeypatch
):
    notes = fake.mkdir("/Notes")
    fake.put("/Notes/a.md", "alpha")
    state: dict = {}
    sync(fake, state, notes)
    before = copy.deepcopy(state)
    fake.put("/Notes/b.md", "bravo")

    lookup = fake._files_get_metadata

    def lookup_then_move(body):
        answer = lookup(body)
        fake.move("/Notes", "/Journal")  # renamed right after Dropbox answered the lookup
        return answer

    monkeypatch.setattr(fake, "_files_get_metadata", lookup_then_move)
    with pytest.raises(DropboxApiError, match="moved while it was listed"):
        sync(fake, state, notes)
    assert state == before

    monkeypatch.setattr(fake, "_files_get_metadata", lookup)
    upserts, deletes, _ = sync(fake, state, notes)
    assert deletes == [] and all(
        row["cognee_structure"]["ancestors"][-1]["name"] == "Journal" for row in upserts.values()
    )


def test_renaming_the_selected_folder_and_a_subfolder_changes_no_content(fake):
    """The path is structure only: a rename re-stages the rows with the same text, so cognee
    keeps every document and extracts nothing again (structure is not hashed)."""
    notes = fake.mkdir("/Notes")
    fake.put("/Notes/a.md", "alpha")
    fake.put("/Notes/Sub/b.md", "bravo")
    state: dict = {}
    before, _, _ = sync(fake, state, notes)

    fake.move("/Notes/Sub", "/Notes/Renamed")
    fake.move("/Notes", "/Journal")
    after, deletes, _ = sync(fake, state, notes)

    assert deletes == []
    assert {k: v["content"] for k, v in after.items()} == {
        k: v["content"] for k, v in before.items()
    }
    assert {v["title"] for v in after.values()} == {v["title"] for v in before.values()}
    names = {tuple(e["name"] for e in v["cognee_structure"]["ancestors"]) for v in after.values()}
    assert names == {("Journal",), ("Renamed", "Journal")}


def test_a_team_space_is_reported_once_and_changes_nothing_else(fake, caplog):
    fake.team_space = True
    a = fake.put("/a.md", "alpha")

    upserts, _, stats = sync(fake, {})

    assert stats["team_space_unsupported"] == 1 and set(upserts) == {a}
    assert caplog.text.count("team space") == 1


def test_a_token_of_another_account_is_refused_before_any_listing(fake):
    fake.put("/a.md", "alpha")
    fake.account = "dbid:someone-else"
    state: dict = {}

    error, stats = run_with_stats(fake, state)

    assert isinstance(error, DropboxAccountMismatchError) and state == {}
    assert not isinstance(error, DropboxAuthError), "hosts must not read it as an expired token"
    assert "files/list_folder" not in fake.calls
    assert (stats["failed"], stats["failed_account"], stats["failed_auth"]) == (1, 1, 0)


# ---------------------------------------------------------------- the date line (D3, round 4)
def first_line(row):
    return row["content"].split("\n", 1)[0]


def test_the_first_line_is_the_day_the_file_was_last_changed(fake):
    md = fake.put("/a.md", "alpha", modified="2025-03-04T23:59:59Z")
    pdf = fake.put("/r.pdf", make_pdf("Report text"), modified="2024-12-31T00:00:01Z")
    info = {"export_as": "markdown"}
    paper = fake.put(
        "/P.paper",
        "# Plan",
        downloadable=False,
        export_info=info,
        modified="2020-01-01T00:00:00Z",
        server_modified="2025-06-07T08:00:00Z",
    )
    bare = fake.put("/b.md", "bravo", modified=None)

    upserts, _, _ = sync(fake, {})

    assert first_line(upserts[md]) == "Modified: 2025-03-04"
    assert first_line(upserts[pdf]) == "Modified: 2024-12-31"
    assert first_line(upserts[paper]) == "Modified: 2025-06-07", "a Paper doc uses server time"
    assert upserts[md]["content"] == "Modified: 2025-03-04\n\nalpha"
    assert upserts[bare]["content"] == "bravo", "no date, no line"


def test_a_move_and_a_folder_rename_keep_the_content(fake):
    notes = fake.mkdir("/Notes")
    a = fake.put("/Notes/Sub/a.md", "alpha", modified="2025-03-04T10:00:00Z")
    state: dict = {}
    before, _, _ = sync(fake, state, notes)

    fake.move("/Notes/Sub/a.md", "/Notes/Other/a.md")
    fake.move("/Notes", "/Journal")
    after, _, _ = sync(fake, state, notes)

    assert after[a]["content"] == before[a]["content"]


def test_the_date_line_moves_only_when_the_bytes_change(fake):
    """The date is the day the content last changed: a re-save of the same bytes, on any day,
    emits nothing; new bytes on a later day carry that day."""
    a = fake.put("/a.md", "alpha", modified="2025-03-04T10:00:00Z")
    state: dict = {}
    first, _, _ = sync(fake, state)

    fake.put("/a.md", "alpha", modified="2025-03-09T18:00:00Z")
    resaved, _, resaved_stats = sync(fake, state)
    fake.put("/a.md", "alpha, edited", modified="2025-03-10T08:00:00Z")
    edited, _, _ = sync(fake, state)

    assert first_line(first[a]) == "Modified: 2025-03-04"
    assert resaved == {} and resaved_stats["unchanged"] == 1
    assert first_line(edited[a]) == "Modified: 2025-03-10"


def test_a_move_of_a_resaved_file_keeps_its_content_day(fake):
    notes = fake.mkdir("/Notes")
    a = fake.put("/Notes/a.md", "alpha", modified="2025-03-04T10:00:00Z")
    state: dict = {}
    sync(fake, state, notes)

    fake.put("/Notes/a.md", "alpha", modified="2025-06-01T10:00:00Z")
    fake.move("/Notes/a.md", "/Notes/Sub/a.md")
    upserts, _, _ = sync(fake, state, notes)

    assert upserts[a]["content"] == "Modified: 2025-03-04\n\nalpha", "same bytes, same day"
    assert upserts[a]["cognee_structure"]["ancestors"][0]["name"] == "Sub"


def test_a_listed_file_without_a_content_hash_is_a_malformed_listing(fake):
    fake.put("/a.md", "alpha")
    del fake.files["/a.md"]["content_hash"]
    state: dict = {}

    with pytest.raises(DropboxApiError, match="malformed file"):
        sync(fake, state)
    assert state == {}


def test_a_render_version_change_lists_whole_and_reemits_every_file(fake):
    a = fake.put("/a.md", "alpha")
    b = fake.put("/b.md", "bravo")
    state: dict = {}
    sync(fake, state)
    assert state["render"] == 1
    unchanged, _, _ = sync(fake, state)
    assert unchanged == {}

    state["render"] = 0  # written by an older rendering
    fake.calls.clear()
    upserts, deletes, _ = sync(fake, state)

    assert "files/list_folder" in fake.calls
    assert set(upserts) == {a, b} and deletes == []
    assert state["render"] == 1


@pytest.mark.parametrize(
    "value", ["2025-13-01T00:00:00Z", "2025-02-30T00:00:00Z", "2025-13-45T00:00:00Z"]
)
def test_an_impossible_date_gives_no_line(fake, value):
    a = fake.put("/a.md", "alpha", modified=value)

    upserts, _, _ = sync(fake, {})

    assert upserts[a]["content"] == "alpha"


# ---------------------------------------------------------------- review round 6
@pytest.mark.parametrize("tag", ["restricted_content", "not_file"])
def test_a_file_dropbox_will_never_serve_is_skipped_not_retried(fake, tag):
    ok = fake.put("/ok.md", "fine")
    blocked = fake.put("/blocked.md", "blocked")
    fake.download_errors[blocked] = lookup_error(tag)
    state: dict = {}

    runs = [sync(fake, state) for _ in range(3)]

    assert set(runs[0][0]) == {ok}
    assert [r[2]["failed"] for r in runs] == [0, 0, 0], "a clean run lets the host retire"
    assert runs[0][2]["skipped_restricted"] == 1 and state["retry"] == []
    assert fake.downloads.count(blocked) == 1


def test_a_download_404_tombstones_a_stored_file_in_the_same_run(fake):
    a = fake.put("/a.md", "alpha")
    state: dict = {}
    sync(fake, state)

    fake.put("/a.md", "alpha, edited")
    fake.download_errors[a] = lookup_error("not_found")
    _, deletes, stats = sync(fake, state)

    assert deletes == [a] and stats["skipped_not_found"] == 1 and stats["deleted"] == 1


def test_one_wait_up_to_the_run_budget_is_slept(fake, waits):
    a = fake.put("/a.md", "alpha")
    fake.queued["files/download"] = [(429, {}, {"Retry-After": "120"})]

    upserts, _, stats = sync(fake, {}, waits=waits)

    assert set(upserts) == {a} and sum(waits) == 120 and stats["failed_rate_limit"] == 0


def test_a_listed_file_without_is_downloadable_is_a_malformed_listing(fake):
    fake.put("/a.md", "alpha")
    del fake.files["/a.md"]["is_downloadable"]
    state: dict = {}

    with pytest.raises(DropboxApiError, match="malformed file"):
        sync(fake, state)
    assert state == {}


# ---------------------------------------------------------------- round 9 (bugs, red first)
def test_a_delta_of_deletes_compares_paths_in_proportion_to_deletes_plus_index(fake, monkeypatch):
    """Each deleted path scanned the whole index: N deletes over M files cost N * M checks."""
    from cognee.tasks.ingestion.connectors import dropbox as dropbox_module

    deletes, files = 200, 20000
    index = {f"/d/f{i:05}.md": f"id:f{i}" for i in range(files)}
    fake.log.extend(
        {
            ".tag": "deleted",
            "name": f"f{i:05}.md",
            "path_lower": f"/d/f{i:05}.md",
            "path_display": f"/d/f{i:05}.md",
        }
        for i in range(deletes)
    )
    state = {
        "v": 1,
        "render": 1,
        "account": ACCOUNT,
        "folder": "",
        "path": "",
        "cursor": "|0",
        "index": index,
        "folders": {"/d": "id:folder-d"},
        "stored": sorted(index.values()),
        "retry": [],
    }
    comparisons = 0
    real_is_under = dropbox_module._is_under

    def counted(path, folder):
        nonlocal comparisons
        comparisons += 1
        return real_is_under(path, folder)

    monkeypatch.setattr(dropbox_module, "_is_under", counted)

    _, removed, stats = sync(fake, state)

    assert len(removed) == deletes and stats["deleted"] == deletes
    assert comparisons < 10 * (deletes + files), f"{comparisons} path comparisons"


def test_a_resave_with_the_same_bytes_on_a_new_day_emits_nothing(fake):
    """Same bytes, new client_modified: the content did not change, so the document must not
    be re-extracted."""
    a = fake.put("/a.md", "alpha", modified="2025-03-04T10:00:00Z")
    state: dict = {}
    sync(fake, state)

    fake.put("/a.md", "alpha", modified="2025-03-09T10:00:00Z")
    upserts, deletes, stats = sync(fake, state)

    assert upserts == {} and deletes == [], f"re-emitted: {upserts.get(a, {}).get('content')!r}"
    assert stats["unchanged"] == 1


# ---------------------------------------------------------------- round 11
def test_new_bytes_with_the_same_text_keep_the_day_and_emit_nothing(fake):
    """A re-save that changes the bytes but not the extracted text (here a UTF-8 BOM, for a PDF
    an annotation or its metadata) must not re-extract the document under a new date."""
    a = fake.put("/a.md", "alpha", modified="2025-03-04T10:00:00Z")
    state: dict = {}
    sync(fake, state)

    fake.put("/a.md", b"\xef\xbb\xbfalpha", modified="2025-03-09T10:00:00Z")
    upserts, deletes, stats = sync(fake, state)

    assert upserts == {} and deletes == [], f"re-emitted: {upserts.get(a, {}).get('content')!r}"
    assert stats["unchanged"] == 1
    fake.put("/a.md", "alpha", modified="2025-03-11T10:00:00Z")
    fake.move("/a.md", "/b.md")
    moved, _, _ = sync(fake, state)
    assert moved[a]["content"] == "Modified: 2025-03-04\n\nalpha", "the text's day is kept"


def test_an_edited_paper_doc_without_a_content_hash_is_emitted_again(fake):
    info = {"export_as": "markdown"}
    paper = fake.put("/P.paper", "# Plan v1", downloadable=False, export_info=info)
    state: dict = {}
    sync(fake, state)

    fake.put("/P.paper", "# Plan v2", downloadable=False, export_info=info)
    upserts, _, _ = sync(fake, state)

    assert upserts[paper]["content"].endswith("# Plan v2")

    fake.put("/P.paper", "# Plan v2", downloadable=False, export_info=info)
    unchanged, _, stats = sync(fake, state)
    assert unchanged == {} and stats["unchanged"] == 1, "same text: nothing emitted"
