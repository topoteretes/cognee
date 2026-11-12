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
(``claim_stored_file``), and removals (``removing_stored_files``) skip a file
another scope claims. The claim and the removal check run under one lock, and a
store that a removal overtook (its file removed between the write and the
claim) stores the file again before claiming it.

Process-local, like the dataset lock: two processes sharing one data root are
not covered.
"""

import threading
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar

from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticLock

# Held by a claim and by a removal from its reference check to its last delete.
_lock = LoopAgnosticLock()
# Guards the counters, which a scope releases without taking ``_lock``.
_counts_guard = threading.Lock()
# Location -> how many open scopes claim it.
_claims: Counter[str] = Counter()
# Bumped by every removal, so a store can tell whether one ran since it wrote.
_removals = 0

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
        with _counts_guard:
            _claims.subtract(own)
            for location in own:
                if _claims[location] <= 0:
                    del _claims[location]


def removals_so_far() -> int:
    """The removal count, read before a store and handed to ``claim_stored_file``."""
    return _removals


async def claim_stored_file(
    location: str, removals_before_store: int, store_again: Callable[[], Awaitable]
) -> None:
    """Claim ``location`` in the open scope; a no-op outside one.

    ``store_again`` writes the file once more. It runs only when a removal ran
    since the store (``removals_before_store``), since that removal may have
    taken the file before this claim could protect it.
    """
    own = _scope.get()
    if own is None:
        return
    async with _lock:
        if _removals != removals_before_store:
            await store_again()
        with _counts_guard:
            _claims[location] += 1
            own[location] += 1


@asynccontextmanager
async def removing_stored_files() -> AsyncIterator[Callable[[str], bool]]:
    """Hold off claims while removing; yields whether a location is claimed elsewhere.

    Run the reference check and the deletes inside the block, and skip every
    location the yielded check reports. The current scope's own claims do not
    count: an operation may remove the files it staged and did not use.
    """
    global _removals
    async with _lock:
        own = _scope.get() or Counter()

        def claimed_elsewhere(location: str) -> bool:
            with _counts_guard:
                return _claims[location] - own[location] > 0

        try:
            yield claimed_elsewhere
        finally:
            _removals += 1
