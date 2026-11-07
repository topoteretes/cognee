"""Pins on stored files — keep cleanup off a file an in-flight ingest will reference.

Stored files are content addressed and shared by every dataset and user under
one data root, and cleanup removes one once no ``Data`` row references it
(``remove_data_file_if_unreferenced``). Ingestion writes the file first and
inserts the referencing row only at its final commit, and a delete in one
dataset holds that dataset's lock while an add into another holds its own, so
nothing orders the two: the reference count can reach zero, and the file be
removed, between the add's write and its commit.

A writer opens :func:`stored_file_pin_scope` around everything from its first
write to its commit; every ``store()`` inside the scope pins its destination
first (:func:`pin_stored_file`), and the scope's exit releases the pins.
Cleanup runs its reference count and removal under :func:`stored_file_removal`,
which skips a pinned file. Both take the same per-file lock, so a pin waits for
a removal of that file already in progress and the write then recreates the
file, and a removal that starts after the pin sees it.

Keys are the backend's absolute location of the file
(``Storage.stored_file_key``), so a writer and a cleanup that reach one file
through different storage roots agree on it.

NOTE: process-local only, like ``dataset_lock`` — workers of one deployment
are not covered. A file pinned by an ingest that then fails stays on disk
unreferenced, as failed-ingest leftovers already do.
"""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from .loop_agnostic_lock import LoopAgnosticLock


class _StoredFileEntry:
    """A file's lock and pin count, plus how many callers are using the entry."""

    __slots__ = ("lock", "pins", "users")

    def __init__(self) -> None:
        self.lock = LoopAgnosticLock()
        self.pins = 0
        self.users = 0


class _PinScope:
    __slots__ = ("closed", "keys")

    def __init__(self) -> None:
        self.keys: list[str] = []
        self.closed = False


# Guarded by a threading.Lock held only for dict and counter updates, never
# while waiting on a file's lock. An entry is dropped once nobody uses it and
# nothing pins it; ``users`` counts callers between fetching the entry and
# releasing its lock, so an entry is never replaced while one is still waiting.
_entries: dict[str, _StoredFileEntry] = {}
_entries_guard = threading.Lock()

# The current writer's scope. ContextVar propagates into child tasks spawned
# with asyncio.create_task, so their stores pin into the same scope.
_pin_scope: ContextVar[_PinScope | None] = ContextVar("stored_file_pin_scope", default=None)


def _release_entry_locked(key: str, entry: _StoredFileEntry) -> None:
    if entry.users == 0 and entry.pins == 0 and _entries.get(key) is entry:
        del _entries[key]


@asynccontextmanager
async def _hold(key: str) -> AsyncIterator[_StoredFileEntry]:
    with _entries_guard:
        entry = _entries.get(key)
        if entry is None:
            entry = _entries[key] = _StoredFileEntry()
        entry.users += 1
    try:
        async with entry.lock:
            yield entry
    finally:
        with _entries_guard:
            entry.users -= 1
            _release_entry_locked(key, entry)


@asynccontextmanager
async def stored_file_pin_scope() -> AsyncIterator[None]:
    """Keep the files stored inside the block pinned until the block exits.

    Nested scopes reuse the outermost one, so the pins live until the
    outermost writer — the one that commits the rows — is done.
    """
    if _pin_scope.get() is not None:
        yield
        return

    scope = _PinScope()
    token = _pin_scope.set(scope)
    try:
        yield
    finally:
        _pin_scope.reset(token)
        with _entries_guard:
            # A task spawned in the scope and still running after it must not
            # pin into it any more: nothing would release that pin.
            scope.closed = True
            for key in scope.keys:
                entry = _entries[key]
                entry.pins -= 1
                _release_entry_locked(key, entry)


async def pin_stored_file(key: str) -> None:
    """Pin ``key`` for the current scope; a no-op outside one.

    Call before writing the file: if a removal of it is in progress this waits
    for it, so the write that follows recreates the file.
    """
    scope = _pin_scope.get()
    if scope is None or scope.closed:
        return

    async with _hold(key) as entry:
        with _entries_guard:
            if scope.closed:
                return
            entry.pins += 1
            scope.keys.append(key)


@asynccontextmanager
async def stored_file_removal(key: str) -> AsyncIterator[bool]:
    """Hold ``key``'s lock; yields whether the file may be removed (it is not pinned).

    Run the reference count and the removal inside the block, so a writer
    cannot pin the file between the two.
    """
    async with _hold(key) as entry:
        with _entries_guard:
            removable = entry.pins == 0
        yield removable
