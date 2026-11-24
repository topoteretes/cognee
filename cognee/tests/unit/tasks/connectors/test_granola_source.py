# Adapted from topoteretes/cloud-backend, CLO-504, cognee-saas-pod/tests/unit/test_granola_source.py at aa05be08.
# Modified to ship in the Cognee SDK: SDK run bounds, include_transcript, reconcile and the scenario trace.
"""The Granola walker, driven with a plain dict for state.

The first half is the Cloud pod's suite, ported; its scripted client answers each request in
turn, and its runs start with a fresh ``reconciled_at`` so the daily reconcile stays out of
their scripts. The second half follows the contract's user scenario trace (one folder, a
personal key) on a fake that answers like Granola's API, then its threats and acceptance list.
"""

import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from cognee.tasks.ingestion.connectors import granola
from cognee.tasks.ingestion.connectors.granola import (
    MAX_NOTE_RETRIES,
    GranolaError,
    _get_note,
    _iter_rows,
    _membership,
    _render,
    granola_source,
)
from cognee.tests.unit.tasks.connectors.fake_granola import FakeGranola, note_id

NOTE = "not_123456789ABCDE"
NOTE2 = "not_ABCDEFGHIJKLMN"
FOLDER = "fol_123456789ABCDE"
NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)
FRESH = NOW.isoformat()


def note(note_id=NOTE):
    return {
        "id": note_id,
        "title": "Planning",
        "summary_text": "Summary",
        "private_notes_text": "DO NOT SHARE",
        "private_notes_markdown": "PRIVATE",
        "transcript": [{"text": "Hello", "speaker": {"diarization_label": "Speaker A"}}],
    }


def page(notes, cursor=None):
    return {"notes": notes, "hasMore": cursor is not None, "cursor": cursor}


class Client:
    """Answers each request with the next scripted response."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, path, params=None, **_kwargs):
        self.calls.append((path, params))
        result = next(self.responses, None)
        assert result is not None, f"unscripted request to {path}"
        if isinstance(result, Exception):
            raise result
        return result


def run(client, state=None, **kwargs):
    state = {} if state is None else state
    state.setdefault("reconciled_at", FRESH)
    stats = {}
    rows = list(_iter_rows(client, FOLDER, state, stats, now=NOW, **kwargs))
    return rows, state, stats


# ---------------------------------------------------------------- the pod's suite, ported


def test_ingests_summary_and_transcript_without_private_scratch_notes():
    rows, state, stats = run(Client([page([{"id": NOTE}]), note()]))
    assert len(rows) == 1
    assert "Summary" in rows[0]["content"] and "Speaker A: Hello" in rows[0]["content"]
    assert "PRIVATE" not in json.dumps(rows) and "DO NOT SHARE" not in json.dumps(rows)
    assert state == {
        "scope": FOLDER,
        "watermark": "2026-11-11T23:55:00+00:00",
        "reconciled_at": FRESH,
        "known": [NOTE],
        "include_transcript": True,
    }
    assert stats["failed"] == 0


def test_cut_mid_page_resumes_without_skipping_or_refetching_completed_note():
    first = Client([page([{"id": NOTE}, {"id": NOTE2}]), note()])
    rows, state, stats = run(first, max_rows=1)
    assert [r["id"] for r in rows] == [NOTE]
    assert "watermark" not in state and stats["failed_budget"] == 1
    second = Client([note(NOTE2)])
    rows, state, stats = run(second, state)
    assert [r["id"] for r in rows] == [NOTE2]
    assert second.calls[0][0] == f"/v1/notes/{NOTE2}"
    assert "pending" not in state and stats["failed"] == 0


def test_incremental_filters_and_pagination():
    client = Client([page([{"id": NOTE}], "next"), note(), page([])])
    rows, _, stats = run(client, {"watermark": "2026-11-01T00:00:00Z"})
    assert len(rows) == 1 and stats["failed"] == 0
    assert client.calls[0][1]["folder_id"] == FOLDER
    assert client.calls[0][1]["updated_after"] == "2026-11-01T00:00:00Z"
    assert client.calls[2][1]["cursor"] == "next"


def test_an_expired_listing_cursor_restarts_from_the_watermark():
    client = Client([GranolaError("expired", 400), page([])])
    _, _, stats = run(client, {"cursor": "expired", "watermark": "2026-11-01"})
    assert "cursor" not in client.calls[1][1]
    assert client.calls[1][1]["updated_after"] == "2026-11-01"
    assert stats["failed"] == 0


def test_deletion_uses_explicit_tombstone_and_a_missing_read_deletes_nothing():
    rows, _, _ = run(Client([page([{"id": NOTE, "deleted_at": "2026-11-01"}])]))
    assert rows == [{"id": NOTE, "_deleted": True}]
    rows, _, stats = run(Client([page([{"id": NOTE}]), GranolaError("missing", 404)]))
    assert rows == [] and stats["failed"] == 1


@pytest.mark.parametrize("status,reason", [(401, "auth"), (429, "rate_limit")])
def test_failure_preserves_pending_progress(status, reason):
    rows, state, stats = run(Client([page([{"id": NOTE}]), GranolaError("refused", status)]))
    assert not rows and "watermark" not in state
    assert state["pending"][0]["id"] == NOTE
    assert stats[f"failed_{reason}"] == 1
    assert "note_failures" not in state


@pytest.mark.parametrize("status", [503, None])
def test_outage_pauses_without_charging_the_note(status):
    # Resumed on a pending page: the note is the run's first request, and the probe fails
    # too, so this is an outage and nobody's retry is used up.
    state = {
        "pending": [{"id": NOTE, "deleted": False, "stamp": None}],
        "next_cursor": None,
        "cycle_start": "2026-11-11T00:00:00+00:00",
    }
    client = Client([GranolaError("down", status), GranolaError("down", status)])
    rows, state, stats = run(client, state)
    assert not rows and state["pending"][0]["id"] == NOTE
    assert stats["failed_unavailable"] == 1 and "failed_source" not in stats
    assert "note_failures" not in state
    assert client.calls[1] == ("/v1/notes", {"page_size": 1})


def member(note_id=NOTE, folders=(), spaces=()):
    return {
        **note(note_id),
        "folder_membership": [dict(folder) for folder in folders],
        "space_membership": [{"id": space} for space in spaces],
    }


def test_note_that_always_fails_with_5xx_does_not_block_later_notes():
    state = {
        "pending": [
            {"id": NOTE, "deleted": False, "stamp": None},
            {"id": NOTE2, "deleted": False, "stamp": None},
        ],
        "next_cursor": None,
        "cycle_start": "2026-11-11T00:00:00+00:00",
    }
    # Run 1: the note 503s, Granola answers the probe, so the note is set aside for a retry
    # by id and the note behind it is imported.
    client = Client([GranolaError("boom", 503), page([]), note(NOTE2)])
    rows, state, stats = run(client, state)
    assert [row["id"] for row in rows] == [NOTE2]
    assert state["note_failures"][NOTE]["attempts"] == 0
    assert state["watermark"] == "2026-11-11T00:00:00+00:00" and "pending" not in state
    # A 5xx never uses up an attempt: retried every cycle, never given up.
    for _ in range(MAX_NOTE_RETRIES + 2):
        client = Client([GranolaError("boom", 503), page([])])
        rows, state, stats = run(client, state)
        assert client.calls[0][0] == f"/v1/notes/{NOTE}"
        assert state["note_failures"][NOTE]["attempts"] == 0
        assert stats["skipped_unavailable"] == 0
    # Once it reads, it is imported and leaves the list.
    rows, state, _ = run(Client([member(folders=[{"id": FOLDER}]), page([])]), state)
    assert [row["id"] for row in rows] == [NOTE] and "note_failures" not in state


def test_an_outage_of_the_note_endpoints_alone_gives_up_nothing():
    """The listing answers, every note GET 503s: nothing is written off."""
    state = {"watermark": "2026-11-01"}
    listing = page([{"id": NOTE}, {"id": NOTE2}])
    client = Client([listing, GranolaError("down", 503), GranolaError("down", 503)])
    _, state, stats = run(client, state)
    for _ in range(3):
        client = Client([GranolaError("down", 503), GranolaError("down", 503), page([])])
        _, state, stats = run(client, state)
    assert set(state["note_failures"]) == {NOTE, NOTE2}
    assert stats["skipped_unavailable"] == 0


def test_retries_by_id_stop_after_a_few_unavailable_notes():
    ids = [f"not_{i:014d}" for i in range(5)]
    state = {
        "watermark": "2026-11-01",
        "note_failures": {i: {"attempts": 0, "stamp": None} for i in ids},
    }
    client = Client([GranolaError("down", 503)] * granola.MAX_UNAVAILABLE_RETRIES + [page([])])
    run(client, state)
    note_calls = [path for path, _ in client.calls if path.startswith("/v1/notes/")]
    assert len(note_calls) == granola.MAX_UNAVAILABLE_RETRIES


def test_a_full_retry_list_pauses_a_5xx_note_instead_of_giving_it_up(monkeypatch):
    monkeypatch.setattr(granola, "MAX_FAILED_NOTES", 1)
    state = {
        "watermark": "2026-11-01",
        "cycle_start": "2026-11-11T00:00:00+00:00",
        "note_failures": {NOTE2: {"attempts": 0, "stamp": None}},
    }
    client = Client([page([{"id": NOTE}]), GranolaError("down", 503)])
    _, state, stats = run(client, state)
    assert state["pending"][0]["id"] == NOTE and stats["failed_unavailable"] == 1
    assert NOTE not in state["note_failures"]


@pytest.mark.parametrize(
    "folders,imported",
    [
        ([{"id": FOLDER, "parent_folder_id": None}], True),
        ([{"id": "fol_CHILD789ABCDEF", "parent_folder_id": FOLDER.upper()}], True),
        ([{"id": "fol_ELSEWHERE12345", "parent_folder_id": None}], False),
        ([], False),
    ],
)
def test_a_retry_by_id_imports_only_a_note_still_in_the_folder(folders, imported):
    state = {
        "watermark": "2026-11-01",
        "note_failures": {NOTE: {"attempts": 1, "stamp": None}},
    }
    rows, state, _ = run(Client([member(folders=folders), page([])]), state)
    assert [row["id"] for row in rows] == ([NOTE] if imported else [])
    assert "note_failures" not in state


def test_a_retry_by_id_checks_the_space_through_membership():
    space = "spa_123456789ABCDE"
    assert _membership(member(spaces=[space]), space) is True
    assert _membership(member(folders=[{"id": FOLDER, "space_id": space}]), space) is True
    assert _membership(member(spaces=["spa_OTHER789ABCDEF"]), space) is False
    assert _membership(note(), space) is None, "no membership in the answer: unknown"
    assert _membership(note(), "all") is True


def test_large_transcript_is_paged_and_admin_disabled_is_summary_only():
    client = Client(
        [
            GranolaError("large", 413),
            note(),
            {"transcript": [{"text": "one"}], "hasMore": True, "cursor": "more"},
            {"transcript": [{"text": "two"}], "hasMore": False, "cursor": None},
        ]
    )
    assert [p["text"] for p in _get_note(client, NOTE)["transcript"]] == ["one", "two"]
    client = Client([GranolaError("large", 413), note(), GranolaError("disabled", 403)])
    assert _get_note(client, NOTE)["transcript"] is None
    # Transcripts disabled may refuse the inline include too: the summary still imports.
    client = Client([GranolaError("disabled", 403), note(), GranolaError("disabled", 403)])
    fetched = _get_note(client, NOTE)
    assert fetched["transcript"] is None and fetched["summary_text"] == "Summary"


def test_429_carries_cooldown_and_resumes_at_the_unread_note():
    cooldown = GranolaError("slow down", 429, retry_after=600)
    rows, state, stats = run(Client([page([{"id": NOTE}, {"id": NOTE2}]), note(), cooldown]))
    assert [row["id"] for row in rows] == [NOTE]
    assert stats["retry_after_seconds"] == 600
    assert stats["failed_rate_limit"] == 1
    rows, _, stats = run(Client([note(NOTE2)]), state)
    assert [row["id"] for row in rows] == [NOTE2]
    assert stats["failed"] == 0


def test_repeated_cursor_cannot_commit_a_watermark():
    rows, state, stats = run(Client([page([], "same"), page([], "same")]))
    assert not rows and stats["failed"] == 1 and "watermark" not in state


def test_replacing_key_reenumerates_historic_notes():
    client = Client([page([])])
    state = {"watermark": "2026-11-01", "cursor": "old", "key_revision": "old"}
    _, state, stats = run(client, state, key_revision="new")
    assert "updated_after" not in client.calls[0][1]
    assert "cursor" not in client.calls[0][1]
    assert state["key_revision"] == "new"
    assert stats["failed"] == 0


def test_disappearing_note_does_not_block_later_notes_or_hold_the_watermark():
    client = Client([page([{"id": NOTE}, {"id": NOTE2}]), GranolaError("gone", 404), note(NOTE2)])
    rows, state, stats = run(client, {"watermark": "2026-11-01"})
    assert [row["id"] for row in rows] == [NOTE2]
    assert state["watermark"] == "2026-11-11T23:55:00+00:00"
    assert state["note_failures"][NOTE]["attempts"] == 1
    assert stats["failed"] == 1


@pytest.mark.parametrize(
    "responses",
    [
        # A 403 on the note while the account still lists (the probe answers): this note's.
        [GranolaError("forbidden", 403), GranolaError("forbidden", 403), page([])],
        [{**note(), "transcript": [{"text": None}]}],
        [note(NOTE2)],
    ],
)
def test_note_specific_errors_do_not_poison_the_queue(responses, monkeypatch):
    monkeypatch.setattr(granola, "MAX_TRANSCRIPT_CHARS", 10)
    client = Client([page([{"id": NOTE}, {"id": NOTE2}]), *responses, note(NOTE2)])
    rows, _, stats = run(client)
    assert [row["id"] for row in rows] == [NOTE2]
    assert stats["failed"] == stats["skipped"] == 1


def test_failed_note_is_retried_by_id_then_given_up_and_an_edit_retries_it():
    state = {"watermark": "2026-11-01"}
    client = Client([page([{"id": NOTE}]), GranolaError("gone", 404)])
    _, state, stats = run(client, state)
    assert state["note_failures"][NOTE]["attempts"] == 1
    for _attempt in (2, 3):
        # Retried directly, then the walk lists only what changed since.
        client = Client([GranolaError("gone", 404), page([])])
        _, state, stats = run(client, state)
        assert client.calls[0][0] == f"/v1/notes/{NOTE}"
        assert stats["failed"] == 1
    assert "note_failures" not in state and stats["skipped_unavailable"] == 1
    # A later edit lists it again with a fresh allowance.
    client = Client([page([{"id": NOTE, "updated_at": "2026-11-12"}]), GranolaError("gone", 404)])
    _, state, _ = run(client, state)
    assert state["note_failures"][NOTE] == {"attempts": 1, "stamp": "2026-11-12"}


def test_retried_note_that_now_reads_is_imported_and_cleared():
    state = {
        "watermark": "2026-11-01",
        "note_failures": {NOTE: {"attempts": 1, "stamp": None}},
    }
    rows, state, stats = run(Client([member(folders=[{"id": FOLDER}]), page([])]), state)
    assert [row["id"] for row in rows] == [NOTE]
    assert "note_failures" not in state and stats["failed"] == 0


def test_full_retry_list_gives_up_a_new_failure_at_once(monkeypatch):
    monkeypatch.setattr(granola, "MAX_FAILED_NOTES", 1)
    state = {
        "watermark": "2026-11-01",
        "cycle_start": "2026-11-11T00:00:00+00:00",
        "note_failures": {NOTE2: {"attempts": 1, "stamp": None}},
    }
    client = Client([page([{"id": NOTE}]), GranolaError("gone", 404)])
    _, state, stats = run(client, state)
    assert NOTE not in state["note_failures"] and stats["skipped_unavailable"] == 1


# ---------------------------------------------------------------- frozen render


SAMPLE = {
    "id": NOTE,
    "title": "Planning",
    "created_at": "2026-11-10T09:00:00Z",
    "summary_markdown": "## Decisions\n- Ship it",
    "summary_text": "Decisions: ship it",
    "private_notes_text": "DO NOT SHARE",
    "web_url": "https://notes.granola.ai/d/abc",
    "transcript": [
        {"text": "Hello", "speaker": {"diarization_label": "Speaker A"}},
        {"text": "Hi", "speaker": {"source": "microphone"}},
        {"text": "Bye"},
    ],
}


def test_the_render_is_byte_identical_to_the_cloud_pod_source():
    """Cloud has synced with this text since v1.6.8-001; a changed byte re-extracts every
    note on the tenant's LLM key. The expected values are the pod's own ``_render`` output at
    cloud-backend aa05be08, copied (the test does not import cloud-backend)."""
    assert json.dumps(_render(dict(SAMPLE), FOLDER)) == (
        '{"id": "not_123456789ABCDE", "title": "Planning", "content": "Planning\\nMeeting date: '
        "2026-11-10T09:00:00Z\\n\\n## Decisions\\n- Ship it\\n\\nTranscript:\\nSpeaker A: "
        'Hello\\nmicrophone: Hi\\nSpeaker: Bye", "_deleted": false, "cognee_node_set": '
        '["granola:fol_123456789ABCDE"], "url": "https://notes.granola.ai/d/abc"}'
    )
    bare = {k: v for k, v in SAMPLE.items() if k not in {"transcript", "summary_markdown"}}
    bare["web_url"] = "http://evil.example/x"
    assert json.dumps(_render(bare, "all")) == (
        '{"id": "not_123456789ABCDE", "title": "Planning", "content": "Planning\\nMeeting date: '
        '2026-11-10T09:00:00Z\\n\\nDecisions: ship it", "_deleted": false, "cognee_node_set": '
        '["granola:all"]}'
    )


# ---------------------------------------------------------------- the user scenario trace

A = "fol_AAAAAAAAAAAAAA"
CHILD = "fol_CCCCCCCCCCCCCC"
B = "fol_BBBBBBBBBBBBBB"


def account(n: int = 3, **kwargs) -> FakeGranola:
    """``n`` notes in folder A (the last in its child folder) and one in folder B."""
    fake = FakeGranola(**kwargs)
    fake.folder(A)
    fake.folder(CHILD, A)
    fake.folder(B)
    for i in range(1, n + 1):
        fake.add(i, folder=CHILD if i == n else A)
    fake.add(99, folder=B)
    fake.tick(hours=1)
    return fake


def sync(fake: FakeGranola, state: dict, scope: str = A, **kwargs):
    stats: dict = {}
    fake.calls.clear()
    rows = list(_iter_rows(fake, scope, state, stats, now=fake.clock, **kwargs))
    fake.tick(hours=1)
    return rows, stats


def ids(rows, deleted=False):
    return sorted(row["id"] for row in rows if row["_deleted"] is deleted)


def test_first_sync_imports_every_summarized_note_of_the_folder_and_its_child_folders():
    fake = account()
    state: dict = {}
    rows, stats = sync(fake, state)

    assert ids(rows) == [note_id(1), note_id(2), note_id(3)]
    assert all(row[granola.dlt_utils.NODE_SET_COLUMN] == [f"granola:{A}"] for row in rows)
    assert note_id(99) not in json.dumps(rows)
    assert state["known"] == [note_id(1), note_id(2), note_id(3)]
    assert stats["failed"] == 0 and stats["scanned"] == 3
    assert stats["requests"] == len(fake.calls)


def test_idle_sync_makes_one_listing_request_and_yields_nothing():
    fake = account()
    state: dict = {}
    sync(fake, state)

    rows, stats = sync(fake, state)

    assert rows == [] and fake.paths() == ["/v1/notes"] and stats["requests"] == 1
    assert stats["failed"] == 0


def test_an_edited_summary_is_rendered_again():
    fake = account()
    state: dict = {}
    sync(fake, state)

    fake.edit(note_id(2), summary_markdown="Revised: we ship Friday")
    rows, _ = sync(fake, state)

    assert ids(rows) == [note_id(2)] and "Revised: we ship Friday" in rows[0]["content"]


def test_a_note_moved_out_of_the_folder_is_tombstoned_by_the_next_reconcile():
    fake = account()
    state: dict = {}
    sync(fake, state)

    fake.move(note_id(1), B)
    rows, _ = sync(fake, state)
    assert rows == [], "the incremental listing does not show it, and nothing deletes yet"

    fake.tick(hours=24)
    rows, stats = sync(fake, state)

    assert rows == [{"id": note_id(1), "_deleted": True}] and stats["deleted"] == 1
    assert note_id(1) not in state["known"]


def test_a_note_deleted_for_a_personal_key_is_tombstoned_by_the_reconcile():
    fake = account()
    state: dict = {}
    sync(fake, state)

    fake.delete(note_id(2))
    fake.tick(hours=24)
    rows, stats = sync(fake, state)

    assert rows == [{"id": note_id(2), "_deleted": True}] and stats["deleted"] == 1
    assert stats["failed"] == 0


def test_deleted_at_from_a_workspace_wide_key_tombstones_at_once():
    fake = account(workspace_wide=True)
    state: dict = {}
    sync(fake, state)

    fake.delete(note_id(3))
    rows, stats = sync(fake, state)

    assert rows == [{"id": note_id(3), "_deleted": True}] and stats["deleted"] == 1
    assert note_id(3) not in state["known"]


def test_an_empty_complete_listing_deletes_nothing_and_reports_failed_access():
    fake = account()
    state: dict = {}
    sync(fake, state)
    known = list(state["known"])

    fake.listing_hidden.update(fake.notes)  # the key was narrowed to nothing
    fake.tick(hours=24)
    rows, stats = sync(fake, state)

    assert rows == [] and stats["deleted"] == 0
    assert stats["failed_access"] == 1 and stats["failed"] == 1
    assert state["known"] == known
    # It lists again on the next run (one request while the listing stays empty).
    rows, stats = sync(fake, state)
    assert rows == [] and stats["failed_access"] == 1


def test_the_first_reconcile_of_a_table_synced_before_it_existed_deletes_nothing():
    """A table synced by the pod source has no ``known``: its first reconcile deletes nothing
    and reads every listed note once, since nothing says the table holds it (an unchanged row
    keeps its identity, so nothing is extracted again)."""
    fake = account()
    state = {"watermark": (fake.clock - timedelta(minutes=5)).isoformat()}

    rows, stats = sync(fake, state)

    assert ids(rows) == [note_id(1), note_id(2), note_id(3)]
    assert stats["deleted"] == 0 and stats["failed"] == 0
    assert state["known"] == [note_id(1), note_id(2), note_id(3)]
    assert "reconciled_at" in state


def test_a_reconcile_listing_cut_by_the_budget_deletes_nothing_and_resumes():
    fake = account(40)  # two reconcile pages of 30
    state: dict = {}
    while "reconciled_at" not in state:
        sync(fake, state)
    fake.delete(note_id(5))
    fake.tick(hours=24)

    rows, stats = sync(fake, state, max_requests=2)  # the idle listing and one page

    assert rows == [] and stats["failed_budget"] == 1
    assert state["reconcile_cursor"] and len(state["reconcile_seen"]) == 30
    rows, stats = sync(fake, state)
    assert rows == [{"id": note_id(5), "_deleted": True}]
    assert "reconcile_seen" not in state and "reconcile_cursor" not in state


def test_a_note_the_walk_emits_while_a_reconcile_listing_is_open_is_never_tombstoned():
    """The listing is newest first, so a note created after it read the first page sits on a
    page already read; the walk's own listing is evidence it is in the scope."""
    fake = account(40)
    state: dict = {}
    while "reconciled_at" not in state:
        sync(fake, state)
    fake.tick(hours=24)
    sync(fake, state, max_requests=2)  # reconcile cut after its first page

    created = fake.add(41, folder=A)
    rows, stats = sync(fake, state)

    assert ids(rows) == [created] and ids(rows, deleted=True) == []
    assert created in state["known"] and stats["deleted"] == 0
    reads = [params for path, params in fake.calls if path == f"/v1/notes/{created}"]
    assert reads == [{"include": "transcript"}], "the walk's read is the evidence; no confirmation"


def test_an_expired_reconcile_cursor_restarts_the_listing_from_the_first_page():
    fake = account(40)
    state: dict = {}
    while "reconciled_at" not in state:
        sync(fake, state)
    fake.tick(hours=24)
    sync(fake, state, max_requests=2)
    fake.expired.add(state["reconcile_cursor"])
    fake.delete(note_id(7))

    rows, stats = sync(fake, state)

    listings = [params for path, params in fake.calls if path == "/v1/notes"]
    assert "cursor" not in listings[-2], "the run lists the scope again from its first page"
    assert rows == [{"id": note_id(7), "_deleted": True}] and stats["failed"] == 0


def test_a_shuffled_reconcile_listing_forgets_and_keeps_the_same_notes():
    """The outcome depends on the set the listing shows, never on its order."""
    outcomes = []
    for seed in (1, 2):
        fake = account(12)
        state: dict = {}
        sync(fake, state)
        fake.delete(note_id(4))
        fake.tick(hours=24)
        listing = fake._list

        def shuffled(params, listing=listing, seed=seed):
            answer = listing(params)
            random.Random(seed).shuffle(answer["notes"])
            return answer

        fake._list = shuffled
        rows, _ = sync(fake, state)
        outcomes.append((rows, state["known"]))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == [{"id": note_id(4), "_deleted": True}]


def test_a_replaced_key_lists_from_the_start_and_deletes_nothing_by_itself():
    fake = account()
    state: dict = {}
    sync(fake, state, key_revision="key-1")
    fake.tick(hours=24)
    sync(fake, state, max_requests=2, key_revision="key-1")  # leave positions behind

    rows, stats = sync(fake, state, key_revision="key-2")

    first = fake.calls[0][1]
    assert "updated_after" not in first and "cursor" not in first
    assert ids(rows, deleted=True) == [] and stats["deleted"] == 0
    assert ids(rows) == [note_id(1), note_id(2), note_id(3)]
    assert state["key_revision"] == "key-2"


def _one_row_then_401(fake: FakeGranola, state: dict, phase: str) -> list[dict]:
    """Set up ``phase`` so the run emits at least one row and then meets a 401; the rows of a
    setup run come back."""
    denied = GranolaError("revoked", 401)
    earlier: list[dict] = []
    if phase == "list":
        fake.fail("list", denied, after=1)
    elif phase == "resumed list":
        fake.fail("list", denied, after=1)
        earlier, _ = sync(fake, state)  # stops on the second page, keeping its cursor
        assert state["cursor"]
        fake.fail("list", denied, after=1)
    elif phase == "note read":
        fake.fail("note", denied, after=1)
    elif phase == "transcript page":
        fake.inline_limit = 30
        fake.edit(note_id(1), transcript=[{"text": "x" * 20}] * 3)
        fake.tick(hours=1)
        fake.fail("transcript", denied, after=1)
    elif phase == "reconcile listing":
        fake.fail("list", denied, after=3)  # the walk's three pages, then the reconcile
    return earlier


@pytest.mark.parametrize(
    "phase", ["list", "resumed list", "note read", "transcript page", "reconcile listing"]
)
def test_a_401_after_rows_keeps_them_and_stops_with_failed_auth_on_every_phase(phase):
    fake = account(25)
    state: dict = {}
    earlier = _one_row_then_401(fake, state, phase)

    rows, stats = sync(fake, state)

    assert ids(rows), "rows emitted before the 401 are kept"
    assert ids(rows, deleted=True) == []
    assert stats["failed_auth"] == 1 and stats["failed"] == 1
    if phase == "reconcile listing":
        assert state["watermark"] and state["reconcile_seen"] == []
    else:
        assert "watermark" not in state
    # The next run, with the key accepted again, finishes without losing a note.
    while "reconciled_at" not in state or "pending" in state:
        more, stats = sync(fake, state)
        rows += more
    assert set(ids(earlier + rows)) == {note_id(i) for i in range(1, 26)}
    assert stats["failed"] == 0


def test_a_429_reports_its_retry_after_and_the_next_run_resumes():
    fake = account()
    fake.fail("note", GranolaError("slow down", 429, retry_after=120), after=1)
    state: dict = {}

    rows, stats = sync(fake, state)
    assert ids(rows) == [note_id(3)]
    assert stats["failed_rate_limit"] == 1 and stats["retry_after_seconds"] == 120

    rows, stats = sync(fake, state)
    assert ids(rows) == [note_id(1), note_id(2)] and stats["failed"] == 0


def test_a_transcript_too_large_to_inline_is_read_page_by_page():
    fake = account(1)
    fake.inline_limit = 10
    fake.edit(note_id(1), transcript=[{"text": f"part {i}"} for i in range(5)])
    state: dict = {}

    rows, _ = sync(fake, state)

    assert rows[0]["content"].endswith("\n".join(f"Speaker: part {i}" for i in range(5)))
    assert fake.paths().count(f"/v1/notes/{note_id(1)}/transcript") == 3


def test_transcripts_disabled_by_the_admin_still_import_the_summary():
    fake = account(1)
    fake.transcripts_disabled = True
    state: dict = {}

    rows, stats = sync(fake, state)

    assert "Summary of meeting 1" in rows[0]["content"]
    assert "Transcript:" not in rows[0]["content"] and stats["failed"] == 0


def test_include_transcript_off_never_asks_for_one_and_renders_without_it():
    fake = account(2)
    state: dict = {}

    rows, _ = sync(fake, state, include_transcript=False)

    assert all("Transcript:" not in row["content"] for row in rows)
    assert not any(path.endswith("/transcript") for path in fake.paths())
    assert not any(params.get("include") for _, params in fake.calls)
    plain = {k: v for k, v in SAMPLE.items() if k != "transcript"}
    assert _render(plain, FOLDER)["content"] == (
        "Planning\nMeeting date: 2026-11-10T09:00:00Z\n\n## Decisions\n- Ship it"
    )
    # A transcript the bare read carries anyway is not rendered.
    client = Client([dict(SAMPLE)])
    assert "transcript" not in _get_note(client, NOTE, include_transcript=False)


def test_an_interrupted_transcript_is_never_published_and_the_note_is_retried():
    fake = account(1)
    fake.inline_limit = 10
    fake.edit(note_id(1), transcript=[{"text": f"part {i}"} for i in range(5)])
    fake.tick(hours=1)
    fake.fail("transcript", GranolaError("down", 503), after=1)
    state: dict = {"reconciled_at": fake.clock.isoformat()}  # the walk alone

    rows, stats = sync(fake, state)

    assert rows == [] and stats["failed"] == 1
    assert note_id(1) in state["note_failures"]
    rows, _ = sync(fake, state)
    assert ids(rows) == [note_id(1)] and "Speaker: part 4" in rows[0]["content"]


def test_private_notes_never_reach_a_row():
    fake = account()
    rows, _ = sync(fake, {})
    assert "PRIVATE" not in json.dumps(rows)


@pytest.mark.parametrize("bound", ["requests", "rows", "deadline"])
def test_a_run_cut_by_a_budget_resumes_without_losing_a_note(bound):
    fake = account(23)
    state: dict = {}
    seconds = [0.0]

    def clock():
        seconds[0] += 1.0
        return seconds[0]

    kwargs = {
        "requests": {"max_requests": 6},
        "rows": {"max_rows": 4},
        "deadline": {"deadline_seconds": 8, "clock": clock},
    }[bound]
    emitted: list[dict] = []
    runs = 0
    for runs in range(1, 41):
        seconds[0] = 0.0
        rows, stats = sync(fake, state, **kwargs)
        emitted += rows
        assert stats["requests"] <= kwargs.get("max_requests", stats["requests"])
        assert len(rows) <= kwargs.get("max_rows", len(rows))
        if "reconciled_at" in state and stats["failed"] == 0:
            break
        assert stats["failed_budget"] == 1
    assert runs > 2, "the bound cut the first runs"
    assert ids(emitted) == sorted(note_id(i) for i in range(1, 24))
    assert ids(emitted, deleted=True) == []


def test_check_active_is_asked_before_every_request():
    fake = account()
    sync(fake, {}, check_active=lambda: None)
    assert fake.check_active_calls == len(fake.calls)


def test_what_check_active_raises_stops_the_run_as_it_is():
    class Revoked(Exception):
        pass

    def revoked():
        raise Revoked

    with pytest.raises(Revoked):
        sync(account(), {}, check_active=revoked)


def test_no_log_line_names_a_note_or_the_scope(caplog):
    fake = account()
    fake.move(note_id(1), B)
    state = {
        "note_failures": {
            note_id(1): {"attempts": 0, "stamp": None},  # read, and found outside the folder
            note_id(2): {"attempts": 2, "stamp": None},  # its third 404: given up
        }
    }
    fake.fail("note", GranolaError("gone", 404), after=1)

    with caplog.at_level("DEBUG"):
        sync(fake, state)

    assert "Gave up" in caplog.text and "left the scope" in caplog.text
    assert "not_" not in caplog.text and A not in caplog.text


def test_the_whole_scenario_replays_on_one_state():
    """first sync, idle, edit, move out, delete, key narrowed, key replaced, revoked key."""
    fake = account(4)
    state: dict = {}
    known = [note_id(i) for i in range(1, 5)]

    rows, _ = sync(fake, state)
    assert ids(rows) == known
    assert sync(fake, state)[0] == []
    fake.edit(note_id(2), summary_markdown="Edited")
    assert ids(sync(fake, state)[0]) == [note_id(2)]

    fake.move(note_id(1), B)
    fake.delete(note_id(3))
    fake.tick(hours=24)
    rows, _ = sync(fake, state)
    assert ids(rows, deleted=True) == [note_id(1), note_id(3)]

    fake.listing_hidden.update(fake.notes)
    fake.tick(hours=24)
    rows, stats = sync(fake, state)
    assert rows == [] and stats["failed_access"] == 1
    fake.listing_hidden.clear()

    rows, stats = sync(fake, state, key_revision="key-2")
    assert ids(rows) == [note_id(2), note_id(4)] and stats["deleted"] == 0

    fake.fail("list", GranolaError("revoked", 401))
    rows, stats = sync(fake, state, key_revision="key-2")
    assert rows == [] and stats["failed_auth"] == 1
    assert state["known"] == [note_id(2), note_id(4)]


# ---------------------------------------------------------------- factory


def test_the_factory_validates_before_any_request():
    named = {"resource_name": "granola_x"}
    with pytest.raises(ValueError, match="folder or space"):
        granola_source(scope="fol_short", api_key="grn_x", **named)
    with pytest.raises(ValueError, match="folder or space"):
        granola_source(scope=["all"], api_key="grn_x", **named)
    with pytest.raises(ValueError, match="service or an api_key"):
        granola_source(scope="all", **named)
    with pytest.raises(ValueError, match="include_transcript"):
        granola_source(scope="all", api_key="grn_x", include_transcript="no", **named)
    for name in ("max_requests", "max_rows", "deadline_seconds"):
        with pytest.raises(ValueError, match=name):
            granola_source(scope="all", api_key="grn_x", **named, **{name: 0})
    with pytest.raises(ValueError, match="resource_name"):
        granola_source(scope="all", api_key="grn_x", resource_name=" ")
    with pytest.raises(ValueError, match="invalid format") as raised:
        granola_source(scope="all", api_key="grn secret", **named)
    assert "secret" not in str(raised.value)


def test_resource_name_is_required_so_two_keys_never_share_a_table():
    """A default named after the scope let two keys syncing ``all`` into datasets of one name
    share a table and its state: the second tombstoned the first's notes (review X2)."""
    with pytest.raises(TypeError, match="resource_name"):
        granola_source(scope="all", api_key="grn_x")


def test_the_factory_names_the_table_from_the_scope_and_tags_the_source():
    source = granola_source(scope="fol_AbCdEfGhIjKlMn", service=FakeGranola(), resource_name="mine")
    assert source.name == "mine"
    assert getattr(source, granola.DOCUMENT_SOURCE_ATTR) == "granola"
    assert getattr(source, granola.dlt_utils.PIPELINE_SCOPE_ATTR) == source.name
    assert set(source.cognee_sync_stats) == {
        "scanned",
        "skipped",
        "deleted",
        "failed",
        "skipped_unavailable",
        "failed_auth",
        "failed_rate_limit",
        "failed_unavailable",
        "failed_budget",
        "failed_access",
        "failed_listing",
        "transcript_too_large",
        "reconcile_unconfirmed",
        "retry_after_seconds",
        "requests",
    }


def test_a_cut_after_a_failed_note_counts_both_in_failed():
    """The pod kept ``max(1, failed)``, which hid the cut behind the note's failure."""
    client = Client(
        [page([{"id": NOTE}, {"id": NOTE2}]), GranolaError("gone", 404), GranolaError("x", 401)]
    )
    _, _, stats = run(client)
    assert stats["failed"] == 2 and stats["failed_auth"] == 1


def test_the_url_column_is_declared_so_a_first_url_changes_no_older_row():
    source = granola_source(scope="all", service=FakeGranola(), resource_name="mine")
    assert source.columns["url"] == {"name": "url", "data_type": "text", "nullable": True}


# ---------------------------------------------------------------- reconcile confirmation


def _reconciled(n: int, **kwargs) -> tuple[FakeGranola, dict]:
    fake = account(n, **kwargs)
    state: dict = {}
    while "reconciled_at" not in state:
        sync(fake, state)
    fake.tick(hours=24)
    return fake, state


def test_a_live_note_the_paged_listing_skipped_is_confirmed_and_kept():
    """The listing pages by offset: a note deleted from a page already read moves the next
    note onto that page, so the second page never shows it. Its own read proves it is live."""
    fake, state = _reconciled(40)
    sync(fake, state, max_requests=2)  # the idle listing and the first reconcile page
    assert len(state["reconcile_seen"]) == 30
    fake.delete(note_id(40))  # on the page already read

    rows, stats = sync(fake, state)

    assert ids(rows, deleted=True) == [] and stats["deleted"] == 0
    assert note_id(10) in state["known"]
    read = [params for path, params in fake.calls if path == f"/v1/notes/{note_id(10)}"]
    assert read == [{}], "confirmed by one read, without the transcript"
    assert "reconcile_candidates" not in state and "reconciled_at" in state


def test_a_candidate_answered_with_deleted_at_or_404_is_tombstoned():
    fake, state = _reconciled(3, workspace_wide=True)
    fake.notes[note_id(1)]["deleted_at"] = fake.clock.isoformat()  # listed, flagged deleted
    del fake.notes[note_id(2)]  # gone: the read answers 404

    rows, _ = sync(fake, state)

    assert ids(rows, deleted=True) == [note_id(1), note_id(2)]
    assert state["known"] == [note_id(3)]


def test_a_401_while_confirming_deletes_nothing_unconfirmed_and_the_next_run_resumes():
    fake, state = _reconciled(4)
    fake.delete(note_id(1))
    fake.delete(note_id(2))
    fake.fail("note", GranolaError("revoked", 401), after=1)

    rows, stats = sync(fake, state)

    assert rows == [{"id": note_id(1), "_deleted": True}]
    assert stats["failed_auth"] == 1
    assert state["reconcile_candidates"] == [note_id(2)]
    rows, stats = sync(fake, state)
    assert rows == [{"id": note_id(2), "_deleted": True}] and stats["failed"] == 0
    assert [path for path, _ in fake.calls] == ["/v1/notes", f"/v1/notes/{note_id(2)}"]
    assert "reconcile_candidates" not in state


def test_a_candidate_the_walk_emits_again_is_no_longer_a_candidate():
    fake, state = _reconciled(3)
    fake.listing_hidden.add(note_id(2))
    fake.fail("note", GranolaError("slow down", 429))
    sync(fake, state)  # listed, then the confirmation read is throttled
    assert state["reconcile_candidates"] == [note_id(2)]
    fake.listing_hidden.clear()
    fake.edit(note_id(2), summary_markdown="Back in view")

    rows, _ = sync(fake, state)

    assert ids(rows) == [note_id(2)] and ids(rows, deleted=True) == []
    assert note_id(2) in state["known"] and "reconcile_candidates" not in state
    reads = [params for path, params in fake.calls if path == f"/v1/notes/{note_id(2)}"]
    assert reads == [{"include": "transcript"}], "the walk's read; no confirmation read"


def test_a_replaced_key_returns_unconfirmed_candidates_to_known():
    fake, state = _reconciled(3)
    fake.delete(note_id(1))
    fake.fail("note", GranolaError("down", 503))
    fake.fail("list", GranolaError("down", 503), after=2)  # an outage: the probe fails too
    sync(fake, state)
    assert state["reconcile_candidates"] == [note_id(1)]

    rows, _ = sync(fake, state, key_revision="key-2")

    # Nothing deleted by the reset; the new key's reconcile confirms the candidate gone.
    assert ids(rows) == [note_id(2), note_id(3)]
    assert ids(rows, deleted=True) == [note_id(1)]
    assert "reconcile_candidates" not in state


# ---------------------------------------------------------------- include_transcript changes


def test_turning_transcripts_off_and_on_renders_every_synced_note_again():
    fake = account(2)
    state: dict = {}
    sync(fake, state)

    rows, stats = sync(fake, state, include_transcript=False)
    assert ids(rows) == [note_id(1), note_id(2)] and stats["deleted"] == 0
    assert all("Transcript:" not in row["content"] for row in rows)

    assert sync(fake, state, include_transcript=False)[0] == [], "then incremental again"

    rows, stats = sync(fake, state, include_transcript=True)
    assert ids(rows) == [note_id(1), note_id(2)] and stats["deleted"] == 0
    assert all("Transcript:" in row["content"] for row in rows)


def test_a_table_synced_before_the_option_existed_is_not_listed_again():
    """The pod always rendered transcripts, so a state without the key means True."""
    fake = account(2)
    state: dict = {}
    sync(fake, state)
    del state["include_transcript"]

    rows, _ = sync(fake, state)

    assert rows == [] and state["include_transcript"] is True


# ---------------------------------------------------------------- review round 1

GRAND = "fol_GGGGGGGGGGGGGG"
ELSEWHERE = "fol_EEEEEEEEEEEEEE"
ELSEWHERE_CHILD = "fol_DDDDDDDDDDDDDD"
ELSEWHERE_GRAND = "fol_FFFFFFFFFFFFFF"


def _deep_account(n: int = 40) -> FakeGranola:
    """``n`` notes in folder A, a grandchild folder under A's child, and a two-level folder
    tree elsewhere."""
    fake = FakeGranola()
    fake.folder(A)
    fake.folder(CHILD, A)
    fake.folder(GRAND, CHILD)
    fake.folder(ELSEWHERE)
    fake.folder(ELSEWHERE_CHILD, ELSEWHERE)
    fake.folder(ELSEWHERE_GRAND, ELSEWHERE_CHILD)
    for i in range(1, n + 1):
        fake.add(i, folder=A)
    return fake


def _skipped_by_the_reconcile_listing(fake: FakeGranola) -> tuple[list[dict], dict]:
    """A reconcile cut after its first page, then a note on that page deleted: the offset
    cursor's second page never shows note 10, whatever folder it sits in."""
    fake.tick(hours=1)
    state: dict = {}
    while "reconciled_at" not in state:
        sync(fake, state)
    fake.tick(hours=24)
    sync(fake, state, max_requests=2)
    fake.delete(note_id(40))
    rows, _ = sync(fake, state)
    return rows, state


@pytest.mark.parametrize("where", ["grandchild folder", "answer without membership"])
def test_a_skipped_candidate_whose_membership_cannot_tell_is_asked_of_the_scope_listing(where):
    fake = _deep_account()
    if where == "grandchild folder":
        fake.move(note_id(10), GRAND, touches=False)
    else:
        fake.no_membership.add(note_id(10))

    rows, state = _skipped_by_the_reconcile_listing(fake)

    assert ids(rows, deleted=True) == [], "a live note Granola lists under A is never forgotten"
    assert note_id(10) in state["known"]
    asked = [p for path, p in fake.calls if path == "/v1/notes" and "created_after" in p]
    assert asked and asked[0]["folder_id"] == A


def test_a_candidate_moved_to_another_folders_grandchild_is_kept_unconfirmed():
    """Its membership cannot place it and the narrow listing does not show it: that absence
    rests on Granola's ``created_at`` filter alone, so the note is kept and counted."""
    fake = _deep_account(3)
    fake.tick(hours=1)
    state: dict = {}
    sync(fake, state)
    fake.move(note_id(2), ELSEWHERE_GRAND)
    fake.tick(hours=24)

    rows, stats = sync(fake, state)

    assert rows == [] and stats["reconcile_unconfirmed"] == 1
    assert note_id(2) in state["known"]


def test_a_live_note_the_narrow_listing_misses_is_kept_not_tombstoned():
    """If the ``created_after``/``created_before`` filter disagrees with the note's own
    ``created_at``, the narrow listing shows nothing for a live note (review F13)."""
    fake, state = _reconciled(3)
    fake.no_membership.add(note_id(2))
    fake.listing_hidden.add(note_id(2))  # every listing, the narrow one included

    rows, stats = sync(fake, state)

    assert rows == [] and stats["deleted"] == 0
    assert stats["reconcile_unconfirmed"] == 1 and note_id(2) in state["known"]
    assert "reconcile_candidates" not in state


def test_a_retry_by_id_of_a_grandchild_folder_note_imports_it():
    fake = _deep_account(2)
    fake.move(note_id(2), GRAND)
    fake.tick(hours=1)
    fake.fail("note", GranolaError("gone", 404))  # its first read fails once
    state: dict = {"reconciled_at": fake.clock.isoformat()}  # the walk alone

    rows, _ = sync(fake, state)
    assert ids(rows) == [note_id(1)] and note_id(2) in state["note_failures"]
    rows, _ = sync(fake, state)

    assert ids(rows) == [note_id(2)] and "note_failures" not in state


def test_a_candidate_refused_alone_is_kept_and_the_reconcile_still_finishes():
    """One note answering 403 while the account lists stopped every reconcile before (R7)."""
    fake, state = _reconciled(4)
    fake.delete(note_id(1))
    fake.listing_hidden.add(note_id(2))
    fake.refused_notes.add(note_id(2))

    rows, stats = sync(fake, state)

    assert rows == [{"id": note_id(1), "_deleted": True}]
    assert note_id(2) in state["known"] and stats["skipped"] == 1 and stats["failed"] == 0
    assert "reconcile_candidates" not in state and "reconciled_at" in state


def test_a_refused_account_mid_walk_stops_with_failed_access_and_charges_no_note():
    fake = account(15)
    state: dict = {}

    def lapse():
        if sum(path.startswith("/v1/notes/") for path, _ in fake.calls) >= 3:
            fake.refuse_account = True

    rows, stats = sync(fake, state, check_active=lapse)

    assert len(ids(rows)) == 3
    assert stats["failed_access"] == 1 and stats["failed"] == 1
    assert "note_failures" not in state, "no note is blamed for the account"
    fake.refuse_account = False
    emitted = ids(rows)
    while "reconciled_at" not in state or "pending" in state:
        more, stats = sync(fake, state)
        emitted += ids(more)
    assert sorted(emitted) == [note_id(i) for i in range(1, 16)]


def test_a_refused_listing_stops_with_failed_access():
    fake = account()
    fake.refuse_account = True

    rows, stats = sync(fake, {})

    assert rows == [] and stats["failed_access"] == 1 and stats["failed_listing"] == 0


def test_a_note_moved_in_without_an_edit_is_read_and_emitted_by_the_reconcile():
    """``updated_after`` never lists a note shared or moved in without an edit (X4a)."""
    fake, state = _reconciled(3)
    moved_in = fake.add(50, folder=B)
    fake.tick(hours=1)
    sync(fake, state)
    fake.move(moved_in, A, touches=False)
    fake.tick(hours=24)

    rows, stats = sync(fake, state)

    assert ids(rows) == [moved_in] and stats["deleted"] == 0
    assert moved_in in state["known"] and "reconcile_new" not in state


def test_a_note_the_reconcile_cannot_read_stays_out_of_known_for_the_next_reconcile():
    fake, state = _reconciled(3)
    moved_in = fake.add(50, folder=B)
    fake.tick(hours=1)
    sync(fake, state)
    fake.move(moved_in, A, touches=False)
    fake.refused_notes.add(moved_in)
    fake.tick(hours=24)

    rows, _ = sync(fake, state)
    assert rows == [] and moved_in not in state["known"]

    fake.refused_notes.clear()
    fake.tick(hours=24)
    rows, _ = sync(fake, state)
    assert ids(rows) == [moved_in]


@pytest.mark.parametrize("inline", [True, False])
def test_a_transcript_over_the_cap_imports_the_summary_and_is_counted(monkeypatch, inline):
    monkeypatch.setattr(granola, "MAX_TRANSCRIPT_CHARS", 10)
    fake = account(1)
    if not inline:
        fake.inline_limit = 5
    fake.edit(note_id(1), transcript=[{"text": "x" * 8}, {"text": "y" * 8}, {"text": "z"}])
    fake.tick(hours=1)
    state: dict = {"reconciled_at": fake.clock.isoformat()}

    rows, stats = sync(fake, state)

    assert ids(rows) == [note_id(1)] and "Summary of meeting 1" in rows[0]["content"]
    assert "Transcript:" not in rows[0]["content"]
    assert stats["transcript_too_large"] == 1 and stats["failed"] == 0


def test_an_inline_answer_over_the_body_cap_reads_the_transcript_page_by_page():
    client = Client(
        [
            granola.GranolaTooLargeError("too large"),
            note(),
            {"transcript": [{"text": "one"}], "hasMore": False, "cursor": None},
        ]
    )
    assert [p["text"] for p in _get_note(client, NOTE)["transcript"]] == ["one"]
    # A bare note over the cap is this note's error.
    client = Client([granola.GranolaTooLargeError("x"), granola.GranolaTooLargeError("x")])
    with pytest.raises(granola.GranolaNoteError):
        _get_note(client, NOTE)


@pytest.mark.parametrize(
    "listing",
    [{"notes": "nope"}, page([{"id": "../users"}]), page([], "same")],
)
def test_a_listing_granola_answers_malformed_stops_with_failed_listing(listing):
    responses = [listing, listing] if listing.get("cursor") == "same" else [listing]
    rows, state, stats = run(Client(responses))
    assert rows == [] and stats["failed_listing"] == 1 and stats["failed_unavailable"] == 0
    assert "watermark" not in state


def test_a_note_the_reconcile_emits_leaves_the_retry_list():
    """The walk set it aside; the same run's reconcile reads it whole, so no retry is left."""
    fake = account(1)
    fake.fail("note", GranolaError("down", 503))
    state: dict = {}

    rows, _ = sync(fake, state)

    assert ids(rows) == [note_id(1)] and "note_failures" not in state


# ---------------------------------------------------------------- review-swarm round (PR 6841)


def test_a_candidate_that_keeps_failing_with_a_5xx_is_kept_and_the_reconcile_finishes():
    """One note answering 500 while Granola otherwise answers stopped every reconcile: the
    candidate stayed first in the queue, so a note deleted after it was never forgotten."""
    fake, state = _reconciled(4)
    fake.listing_hidden.add(note_id(1))
    fake.unavailable_notes.add(note_id(1))
    fake.delete(note_id(3))

    rows, stats = sync(fake, state)

    assert rows == [{"id": note_id(3), "_deleted": True}]
    assert note_id(1) in state["known"] and stats["skipped"] == 1
    assert stats["failed_unavailable"] == 0
    assert "reconcile_candidates" not in state and "reconciled_at" in state


def test_a_new_note_that_keeps_failing_with_a_5xx_waits_for_the_next_reconcile():
    fake, state = _reconciled(3)
    moved_in = fake.add(50, folder=B)
    fake.tick(hours=1)
    sync(fake, state)
    fake.move(moved_in, A, touches=False)
    fake.unavailable_notes.add(moved_in)
    fake.tick(hours=24)

    rows, stats = sync(fake, state)
    assert rows == [] and moved_in not in state["known"] and stats["failed_unavailable"] == 0
    assert "reconcile_new" not in state

    fake.unavailable_notes.clear()
    fake.tick(hours=24)
    rows, _ = sync(fake, state)
    assert ids(rows) == [moved_in]


def test_a_candidate_5xx_during_an_outage_still_stops_the_run_and_keeps_the_queue():
    fake, state = _reconciled(3)
    fake.listing_hidden.add(note_id(1))
    fake.unavailable_notes.add(note_id(1))
    fake.fail("list", GranolaError("down", 503), after=2)  # the walk and the reconcile list

    rows, stats = sync(fake, state)

    assert rows == [] and stats["failed_unavailable"] == 1
    assert state["reconcile_candidates"] == [note_id(1)]


def test_a_state_written_for_another_scope_is_refused():
    """Two scopes on one resource_name shared one state: the second scope's reconcile
    tombstoned every note of the first."""
    fake = account(2)
    state: dict = {}
    sync(fake, state, scope=A)
    known = list(state["known"])

    with pytest.raises(ValueError, match="another Granola scope"):
        sync(fake, state, scope=B)
    assert state["known"] == known


def test_a_state_from_before_the_scope_was_stored_takes_the_scope():
    fake = account(2)
    state: dict = {}
    sync(fake, state, scope=A)
    del state["scope"]
    sync(fake, state, scope=A)
    assert state["scope"] == A


@pytest.mark.parametrize(
    "scope",
    [
        "fol_ſAAAAAAAAAAAAA",  # LATIN SMALL LETTER LONG S matches [a-z] under IGNORECASE
        "fol_KAAAAAAAAAAAAA",  # KELVIN SIGN
        "FOL_AAAAAAAAAAAAAA",  # a second node set for the same folder
        "Spa_AAAAAAAAAAAAAA",
    ],
)
def test_scope_ids_are_ascii_with_a_lowercase_prefix(scope):
    with pytest.raises(ValueError, match="folder or space"):
        granola.validate_scope(scope)


def test_a_note_id_outside_ascii_is_refused_from_a_listing():
    with pytest.raises(GranolaError):
        granola._note_id({"id": "not_KAAAAAAAAAAAAA"})


def test_a_retried_note_whose_scope_listing_repeats_a_cursor_is_charged():
    """A repeated cursor on the note's own narrow listing was taken for an outage: never
    charged, so the note was retried every cycle for ever."""
    unplaced = {**note(), "created_at": "2026-11-10T09:00:00Z"}  # no membership to place it
    client = Client([unplaced, page([], "c"), page([], "c"), page([])])
    _, state, stats = run(client, {"note_failures": {NOTE: {"attempts": 0, "stamp": None}}})
    assert client.calls[1][1]["created_after"] == "2026-11-10T08:59:59Z"
    assert state["note_failures"][NOTE]["attempts"] == 1
    assert stats["failed_unavailable"] == 0 and stats["failed_listing"] == 0


def test_max_rows_bounds_the_reconcile_tombstones_too():
    fake, state = _reconciled(4)
    for n in (1, 2, 3):
        fake.delete(note_id(n))

    rows, stats = sync(fake, state, max_rows=1)
    assert len(rows) == 1 and stats["failed_budget"] == 1

    rows, _ = sync(fake, state)
    assert len(ids(rows, deleted=True)) == 2 and "reconcile_candidates" not in state


def test_a_note_403_while_the_key_is_revoked_stops_with_failed_auth_and_charges_nothing():
    fake = account(2)
    state: dict = {"reconciled_at": fake.clock.isoformat()}
    sync(fake, state)
    fake.edit(note_id(1), title="Renamed")
    fake.refused_notes.add(note_id(1))
    fake.fail("list", GranolaError("revoked", 401), after=1)  # the probe after the 403

    _, stats = sync(fake, state)

    assert stats["failed_auth"] == 1 and "note_failures" not in state


def test_a_transcript_over_the_cap_counts_only_a_note_that_is_imported(monkeypatch):
    monkeypatch.setattr(granola, "MAX_TRANSCRIPT_CHARS", 10)
    fake = account(1)
    fake.edit(note_id(1), transcript=[{"text": "x" * 20}])
    fake.move(note_id(1), B)
    state: dict = {
        "reconciled_at": fake.clock.isoformat(),
        "note_failures": {note_id(1): {"attempts": 0, "stamp": None}},
    }

    rows, stats = sync(fake, state)

    assert ids(rows) == [] and stats["transcript_too_large"] == 0


def test_a_note_without_a_summary_is_imported_once_granola_writes_one():
    fake = account(1)
    pending = fake.add(7, folder=A, summarized=False)
    fake.tick(hours=1)
    state: dict = {}

    rows, _ = sync(fake, state)
    assert pending not in ids(rows)

    fake.edit(pending, summary_markdown="Summary written later")
    rows, _ = sync(fake, state)
    assert ids(rows) == [pending] and "Summary written later" in rows[0]["content"]


def test_another_scope_on_one_resource_name_stops_the_extract_with_a_value_error(tmp_path):
    """Through dlt the refusal arrives wrapped in PipelineStepFailed, the ValueError its cause."""
    import dlt
    from dlt.pipeline.exceptions import PipelineStepFailed

    fake = account(1)
    pipeline = dlt.pipeline("granola_scope", dataset_name="d", pipelines_dir=str(tmp_path))
    pipeline.extract(granola_source(scope=A, resource_name="shared", service=fake))

    with pytest.raises(PipelineStepFailed) as raised:
        pipeline.extract(granola_source(scope=B, resource_name="shared", service=fake))

    causes = []
    error: BaseException | None = raised.value
    while error is not None:
        causes.append(error)
        error = error.__cause__ or error.__context__
    assert any(
        isinstance(cause, ValueError) and "another Granola scope" in str(cause) for cause in causes
    )
