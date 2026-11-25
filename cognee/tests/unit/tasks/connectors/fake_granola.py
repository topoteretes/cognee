"""Granola's public API (docs.granola.ai, v1.7.0) in memory, behind ``GranolaClient.get``.

Listings are newest first by ``created_at``, filtered by ``folder_id`` (every descendant folder
included), ``space_id``, ``updated_after`` and ``created_after`` / ``created_before``
(exclusive), and paged by an offset cursor that the fake can expire. A note is read with
``include=transcript`` (413 above ``inline_limit`` characters; ``transcript: null`` while
transcripts are disabled) or through the paged transcript endpoint (403 while disabled). Its
``folder_membership`` names its own folder and that folder's parent only. A deletion seen by a
workspace-wide key keeps the note listed with ``deleted_at``; any other deletion removes it.
Only a note with a generated summary is listed; ``add(summarized=False)`` makes one without.
``refuse_account`` answers every request 403 (subscription inactive, IP allowlist); notes in
``refused_notes`` answer their reads 403, notes in ``unavailable_notes`` 500; notes in
``no_membership`` are read without ``folder_membership`` or ``space_membership``.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from cognee.tasks.ingestion.connectors.granola import GranolaError

START = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)


def note_id(n: int) -> str:
    return f"not_{n:014d}"


class FakeGranola:
    def __init__(self, *, workspace_wide: bool = False):
        self.clock = START
        self.workspace_wide = workspace_wide
        self.notes: dict[str, dict[str, Any]] = {}
        self.folders: dict[str, str | None] = {}  # folder id -> parent folder id
        self.calls: list[tuple[str, dict]] = []
        self.failures: list[list] = []  # [kind, error, requests of that kind to let through]
        self.transcripts_disabled = False
        self.refuse_account = False
        self.refused_notes: set[str] = set()
        self.unavailable_notes: set[str] = set()
        self.no_membership: set[str] = set()
        self.inline_limit: int | None = None
        self.transcript_page = 2
        self.expired: set[str] = set()
        self.listing_hidden: set[str] = set()  # visible by id, absent from every listing
        self.check_active_calls = 0
        self._issued: dict[str, int] = {}

    # -- the account -------------------------------------------------------------------------
    def tick(self, **delta: float) -> datetime:
        self.clock += timedelta(**delta) if delta else timedelta(minutes=1)
        return self.clock

    def folder(self, folder_id: str, parent: str | None = None) -> None:
        self.folders[folder_id] = parent

    def add(
        self,
        n: int,
        *,
        folder: str | None = None,
        space: str | None = None,
        title: str | None = None,
        summary: str | None = None,
        transcript: list[dict] | None = None,
        summarized: bool = True,
    ) -> str:
        self.tick()
        nid = note_id(n)
        self.notes[nid] = {
            "id": nid,
            "title": title or f"Meeting {n}",
            "created_at": self.clock.isoformat(),
            "updated_at": self.clock.isoformat(),
            "deleted_at": None,
            "summary_markdown": (summary or f"Summary of meeting {n}") if summarized else None,
            "private_notes_text": f"PRIVATE scratch of meeting {n}",
            "web_url": f"https://notes.granola.ai/d/{nid}",
            "folder": folder,
            "space": space,
            "transcript": transcript
            if transcript is not None
            else [{"text": f"Line of meeting {n}", "speaker": {"diarization_label": "Speaker A"}}],
        }
        return nid

    def edit(self, nid: str, **fields: Any) -> None:
        self.tick()
        self.notes[nid].update(fields, updated_at=self.clock.isoformat())

    def move(self, nid: str, folder: str | None, *, touches: bool = True) -> None:
        self.tick()
        self.notes[nid]["folder"] = folder
        if touches:
            self.notes[nid]["updated_at"] = self.clock.isoformat()

    def delete(self, nid: str) -> None:
        self.tick()
        if self.workspace_wide:
            self.notes[nid].update(deleted_at=self.clock.isoformat())
            self.notes[nid]["updated_at"] = self.clock.isoformat()
        else:
            del self.notes[nid]

    def fail(self, kind: str, error: GranolaError, *, after: int = 0) -> None:
        """Answer a request of ``kind`` (``list``, ``note`` or ``transcript``) with ``error``,
        once, after letting ``after`` requests of that kind through."""
        assert kind in {"list", "note", "transcript"}
        self.failures.append([kind, error, after])

    # -- the API -----------------------------------------------------------------------------
    def get(self, path: str, params: dict | None = None, *, check_active=None, time_left=None):
        if check_active is not None:
            self.check_active_calls += 1
            check_active()
        params = dict(params or {})
        self.calls.append((path, params))
        if self.refuse_account:
            raise GranolaError("forbidden", 403)
        kind = (
            "list"
            if path == "/v1/notes"
            else "transcript"
            if path.endswith("/transcript")
            else "note"
        )
        failure = next((f for f in self.failures if f[0] == kind), None)
        if failure is not None and failure[2]:
            failure[2] -= 1
        elif failure is not None:
            self.failures.remove(failure)
            raise failure[1]
        if path == "/v1/notes":
            return self._list(params)
        if path.endswith("/transcript"):
            return self._transcript(path.split("/")[3], params)
        return self._note(path.split("/")[3], params)

    def _in(self, note: dict, params: dict) -> bool:
        folder_id = params.get("folder_id")
        if folder_id is not None and folder_id not in self._chain(note["folder"]):
            return False
        created = datetime.fromisoformat(note["created_at"])
        for bound, outside in (
            ("created_after", lambda edge: created <= edge),
            ("created_before", lambda edge: created >= edge),
        ):
            if params.get(bound) is not None and outside(_time(params[bound])):
                return False
        if params.get("space_id") is not None and note["space"] != params["space_id"]:
            return False
        after = params.get("updated_after")
        if after is not None and datetime.fromisoformat(note["updated_at"]) <= (
            datetime.fromisoformat(after)
        ):
            return False
        return note["summary_markdown"] is not None and note["id"] not in self.listing_hidden

    def _chain(self, folder: str | None) -> list[str]:
        """The folder and every ancestor of it."""
        chain = []
        while folder is not None:
            chain.append(folder)
            folder = self.folders.get(folder)
        return chain

    def _list(self, params: dict) -> dict:
        cursor = params.get("cursor")
        if cursor in self.expired:
            raise GranolaError("expired", 400)
        offset = self._issued[cursor] if cursor else 0
        size = params.get("page_size", 10)
        assert 1 <= size <= 30, "Granola pages 1 to 30 notes"
        listed = sorted(
            (n for n in self.notes.values() if self._in(n, params)),
            key=lambda n: n["created_at"],
            reverse=True,
        )
        page = listed[offset : offset + size]
        more = offset + size < len(listed)
        token = None
        if more:
            token = f"cur{len(self._issued)}"
            self._issued[token] = offset + size
        keys = ("id", "title", "created_at", "updated_at", "deleted_at")
        return {
            "notes": [{key: n[key] for key in keys} for n in page],
            "hasMore": more,
            "cursor": token,
        }

    def _note(self, nid: str, params: dict) -> dict:
        note = self.notes.get(nid)
        if note is None:
            raise GranolaError("missing", 404)
        if nid in self.refused_notes:
            raise GranolaError("forbidden", 403)
        if nid in self.unavailable_notes:
            raise GranolaError("down", 500)
        answer = {
            key: note[key]
            for key in (
                "id",
                "title",
                "created_at",
                "updated_at",
                "deleted_at",
                "summary_markdown",
                "private_notes_text",
                "web_url",
            )
        }
        folder = note["folder"]
        answer["folder_membership"] = (
            [{"id": folder, "parent_folder_id": self.folders.get(folder), "space_id": None}]
            if folder
            else []
        )
        answer["space_membership"] = [{"id": note["space"]}] if note["space"] else []
        if nid in self.no_membership:
            del answer["folder_membership"], answer["space_membership"]
        if params.get("include") == "transcript":
            if self.transcripts_disabled:
                answer["transcript"] = None
                return answer
            size = sum(len(part["text"]) for part in note["transcript"])
            if self.inline_limit is not None and size > self.inline_limit:
                raise GranolaError("too large", 413)
            answer["transcript"] = note["transcript"]
        return answer

    def _transcript(self, nid: str, params: dict) -> dict:
        if self.transcripts_disabled:
            raise GranolaError("disabled", 403)
        parts = self.notes[nid]["transcript"]
        offset = int(params["cursor"].removeprefix("t")) if params.get("cursor") else 0
        page = parts[offset : offset + self.transcript_page]
        more = offset + self.transcript_page < len(parts)
        cursor = f"t{offset + self.transcript_page}" if more else None
        return {"transcript": page, "hasMore": more, "cursor": cursor}

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
