# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""An Evernote account that follows the sync API: every change takes the next update sequence
number, ``getFilteredSyncChunk(after_usn, max_entries)`` returns what changed after it in USN order
with ``chunkHighUSN`` the last USN it holds (unset when nothing is left), and ``getSyncState``
reports ``updateCount``, ``currentTime`` and ``fullSyncBefore``.

It stands in for the connector's ``service`` (no Thrift involved): each method takes
``check_active`` and calls it first, as the real client does before each attempt.
"""

from types import SimpleNamespace
from typing import Any

from cognee.tasks.ingestion.connectors.evernote import NoteNotFoundError


class FakeEvernote:
    def __init__(self, user_id: int = 1001) -> None:
        self.user_id = user_id
        self.update_count = 0
        self.current_time = 1_700_000_000_000
        self.full_sync_before = 0
        self.notebooks: dict[str, dict] = {}
        self.tags: dict[str, dict] = {}
        self.notes: dict[str, dict] = {}
        self.expunged: list[tuple[str, str, int]] = []  # (kind, guid, usn)
        self.calls: list[tuple] = []
        self._failures: list[dict] = []

    # -- the account -------------------------------------------------------
    def _usn(self) -> int:
        self.update_count += 1
        self.current_time += 1000
        return self.update_count

    def add_notebook(self, guid: str, name: str) -> None:
        self.notebooks[guid] = {"name": name, "usn": self._usn()}

    def rename_notebook(self, guid: str, name: str) -> None:
        self.add_notebook(guid, name)

    def add_tag(self, guid: str, name: str) -> None:
        self.tags[guid] = {"name": name, "usn": self._usn()}

    def rename_tag(self, guid: str, name: str) -> None:
        self.add_tag(guid, name)

    def add_note(
        self, guid: str, content: str, *, title: str = "", notebook: str = "nb1", tags=()
    ) -> None:
        self.notes[guid] = {
            "title": title or f"Note {guid}",
            "content": f"<en-note><div>{content}</div></en-note>",
            "notebook": notebook,
            "tags": list(tags),
            "active": True,
            "usn": self._usn(),
        }

    def edit_note(self, guid: str, content: str | None = None, **fields: Any) -> None:
        note = self.notes[guid]
        if content is not None:
            note["content"] = f"<en-note><div>{content}</div></en-note>"
        note.update(fields)
        note["usn"] = self._usn()

    def trash_note(self, guid: str) -> None:
        self.edit_note(guid, active=False)

    def expunge_note(self, guid: str) -> None:
        del self.notes[guid]
        self.expunged.append(("note", guid, self._usn()))

    def expunge_tag(self, guid: str) -> None:
        """Evernote takes the tag off its notes; the chunk names the tag as expunged."""
        del self.tags[guid]
        for note in self.notes.values():
            if guid in note["tags"]:
                note["tags"].remove(guid)
        self.expunged.append(("tag", guid, self._usn()))

    def expunge_notebook(self, guid: str) -> None:
        """Evernote expunges the notebook's notes with it; the chunk names only the notebook."""
        del self.notebooks[guid]
        for note_guid in [g for g, n in self.notes.items() if n["notebook"] == guid]:
            del self.notes[note_guid]
        self.expunged.append(("notebook", guid, self._usn()))

    def drop_note_silently(self, guid: str) -> None:
        """Gone without an expunge record, as after an account restore."""
        del self.notes[guid]

    # -- failures ----------------------------------------------------------
    def fail(
        self, method: str, error: Exception, *, call: int = 1, times: int = 1, guid=None
    ) -> None:
        """Raise ``error`` from the ``call``-th call of ``method`` on, ``times`` times; with
        ``guid``, only on calls for that note."""
        self._failures.append(
            {"method": method, "error": error, "call": call, "times": times, "guid": guid}
        )

    def _check(self, method: str, args: tuple) -> None:
        number = sum(1 for c in self.calls if c[0] == method)
        for failure in self._failures:
            if failure["guid"] is not None and failure["guid"] not in args:
                continue
            if failure["method"] == method and failure["times"] and number >= failure["call"]:
                failure["times"] -= 1
                raise failure["error"]

    def _enter(self, method: str, check_active, *args: Any) -> None:
        if check_active is not None:
            check_active()
        self.calls.append((method, *args))
        self._check(method, args)

    def calls_to(self, method: str) -> list[tuple]:
        return [c[1:] for c in self.calls if c[0] == method]

    # -- the service interface ---------------------------------------------
    def get_user_id(self, check_active=None) -> int:
        self._enter("get_user_id", check_active)
        return self.user_id

    def get_sync_state(self, check_active=None) -> Any:
        self._enter("get_sync_state", check_active)
        return SimpleNamespace(
            currentTime=self.current_time,
            fullSyncBefore=self.full_sync_before,
            updateCount=self.update_count,
        )

    def get_filtered_sync_chunk(self, after_usn: int, max_entries: int, check_active=None):
        self._enter("get_filtered_sync_chunk", check_active, after_usn, max_entries)
        entries = [("notebook", g, n["usn"]) for g, n in self.notebooks.items()]
        entries += [("tag", g, t["usn"]) for g, t in self.tags.items()]
        entries += [("note", g, n["usn"]) for g, n in self.notes.items()]
        entries += [(f"expunged_{kind}", g, usn) for kind, g, usn in self.expunged]
        entries = sorted((e for e in entries if e[2] > after_usn), key=lambda e: e[2])
        entries = entries[:max_entries]
        chunk = SimpleNamespace(
            chunkHighUSN=entries[-1][2] if entries else None,
            updateCount=self.update_count,
            currentTime=self.current_time,
            notes=[],
            notebooks=[],
            tags=[],
            expungedNotes=[],
            expungedNotebooks=[],
            expungedTags=[],
        )
        # Thrift leaves an empty list field unset: None, never [].
        for kind, guid, _usn in entries:
            if kind == "notebook":
                chunk.notebooks.append(
                    SimpleNamespace(guid=guid, name=self.notebooks[guid]["name"])
                )
            elif kind == "tag":
                chunk.tags.append(SimpleNamespace(guid=guid, name=self.tags[guid]["name"]))
            elif kind == "note":
                note = self.notes[guid]
                chunk.notes.append(
                    SimpleNamespace(
                        guid=guid,
                        title=note["title"],
                        notebookGuid=note["notebook"],
                        tagGuids=list(note["tags"]) or None,
                        updateSequenceNum=note["usn"],
                        active=note["active"],
                        deleted=None if note["active"] else self.current_time,
                    )
                )
            elif kind == "expunged_note":
                chunk.expungedNotes.append(guid)
            elif kind == "expunged_notebook":
                chunk.expungedNotebooks.append(guid)
            elif kind == "expunged_tag":
                chunk.expungedTags.append(guid)
        for field in (
            "notes",
            "notebooks",
            "tags",
            "expungedNotes",
            "expungedNotebooks",
            "expungedTags",
        ):
            setattr(chunk, field, getattr(chunk, field) or None)
        return chunk

    def get_note(self, guid: str, check_active=None) -> Any:
        """``getNote`` with content: the note's current title, notebook, tags and ENML."""
        self._enter("get_note", check_active, guid)
        if guid not in self.notes:
            raise NoteNotFoundError("Evernote note not found")
        note = self.notes[guid]
        return SimpleNamespace(
            guid=guid,
            title=note["title"],
            content=note["content"],
            notebookGuid=note["notebook"],
            tagGuids=list(note["tags"]) or None,
        )


def account(notes: int = 3, *, user_id: int = 1001) -> FakeEvernote:
    """One notebook ``Work`` (``nb1``), one ``Home`` (``nb2``), tags ``urgent`` and ``later``, and
    ``notes`` notes ``n1``..``nN`` in Work, each with the text ``Body of nK``."""
    fake = FakeEvernote(user_id=user_id)
    fake.add_notebook("nb1", "Work")
    fake.add_notebook("nb2", "Home")
    fake.add_tag("t1", "urgent")
    fake.add_tag("t2", "later")
    for number in range(1, notes + 1):
        fake.add_note(f"n{number}", f"Body of n{number}")
    return fake
