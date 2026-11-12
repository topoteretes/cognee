"""Keep a stored file from being removed while an operation is about to use it.

Stored files are content-addressed and shared: identical bytes under one name
land under one key, for every dataset of a tenant. A file is removed once no
``Data`` row points at it, but a row points at its file only once it commits,
and an add (or an update's staged content) stores the file well before that.
Without a claim, a removal in another dataset (whose dataset lock is not this
one) could read "no row" in between and delete the file an in-flight add is
about to commit, leaving that row pointing at nothing.

So every operation that stores files for rows it has yet to commit holds them
in a scope (``holding_stored_files``); each store claims its file in it
(``store_claimed``), and a removal (``removing_stored_files``) leaves out a
file another scope claims.

Everything is per location. A removal marks the locations it takes, then runs
its reference check and deletes with no shared lock held; only a claim on one
of those locations waits for it. A store a removal overtook (its location
removed after the store began, before the claim) writes the file again; a
removal of any other location does not make it.

Process-local, like the dataset lock: two processes sharing one data root are
not covered.
"""

import threading
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar

from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticEvent

# Guards the state below. Held only for dict updates, never across I/O.
_guard = threading.Lock()
# Location -> how many open scopes claim it.
_claims: Counter[str] = Counter()
# Location -> the event its removal sets once its delete is over.
_removing: dict[str, LoopAgnosticEvent] = {}
# Location -> the sequence number of its last removal, kept while a store that
# began before it is in flight.
_removed_at: dict[str, int] = {}
# Sequence number each in-flight store began at -> how many began there.
_in_flight: Counter[int] = Counter()
_sequence = 0

_scope: ContextVar[Counter[str] | None] = ContextVar("stored_file_claims", default=None)


@asynccontextmanager
async def holding_stored_files() -> AsyncIterator[None]:
    """Claim the files stored inside this block until it ends.

    Opened around the work that stores files and commits the rows that point at
    them. A nested block joins the open one, so its claims last until the
    outermost block ends.
    """
    if _scope.get() is not None:
        yield
        return
    own: Counter[str] = Counter()
    token = _scope.set(own)
    try:
        yield
    finally:
        _scope.reset(token)
        with _guard:
            _claims.subtract(own)
            for location in own:
                if _claims[location] <= 0:
                    del _claims[location]


async def store_claimed(store: Callable[[], Awaitable[str]]) -> str:
    """Run ``store`` and claim the location it returns in the open scope.

    ``store`` writes the file and returns its location; it runs a second time
    when a removal took that location after the store began. Outside a scope
    it only stores.
    """
    own = _scope.get()
    if own is None:
        return await store()

    with _guard:
        began = _sequence
        _in_flight[began] += 1
    try:
        location = await store()
        with _guard:
            _claims[location] += 1
            own[location] += 1
            removal = _removing.get(location)
        if removal is not None:
            # A removal took the location before this claim; its delete may
            # still be running. Wait for it, then store the file again below.
            await removal.wait()
        with _guard:
            overtaken = _removed_at.get(location, began) > began
        if overtaken:
            await store()
        return location
    finally:
        with _guard:
            _in_flight[began] -= 1
            if _in_flight[began] <= 0:
                del _in_flight[began]
            _forget_old_removals()


@asynccontextmanager
async def removing_stored_files(locations: list[str]) -> AsyncIterator[list[str]]:
    """Take the ``locations`` no other scope claims; yields them, to check and remove.

    Run the reference check and the deletes inside the block, on the yielded
    locations only. A location another removal is already taking is left to
    it. The current scope's own claims do not count: an operation may remove
    the files it staged and did not use.
    """
    global _sequence
    own = _scope.get() or Counter()
    taken: dict[str, LoopAgnosticEvent] = {}
    with _guard:
        for location in dict.fromkeys(locations):
            if _claims[location] - own[location] <= 0 and location not in _removing:
                taken[location] = _removing[location] = LoopAgnosticEvent()

    try:
        yield list(taken)
    finally:
        with _guard:
            _sequence += 1
            for location in taken:
                _removed_at[location] = _sequence
                del _removing[location]
            _forget_old_removals()
        for done in taken.values():
            done.set()


def _forget_old_removals() -> None:
    """Drop removals no in-flight store began before. Call with ``_guard`` held."""
    if not _in_flight:
        _removed_at.clear()
        return
    oldest = min(_in_flight)
    for location in [loc for loc, at in _removed_at.items() if at <= oldest]:
        del _removed_at[location]
