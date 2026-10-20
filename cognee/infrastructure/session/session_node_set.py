"""A session's pinned ``node_set``, carried into the graph by ``improve()``.

A typed QA or trace entry may carry ``node_set`` (``cognee/memory/entries.py``),
the same node-set names ``add()`` takes and ``recall(node_name=...)`` filters
on; the call-level ``remember(..., session_id=..., node_set=[...])`` is the same
thing. The first write that carries one pins it on the session; later writes
must repeat it or omit it, and a different set is rejected with
``SessionNodeSetConflictError`` (HTTP 409), so one session is never split
across node sets. Identify the conflict by that error name, not the status: the
remember routes answer other failures with 409 too. The pinned set lives in an
internal session-context row, like the improve watermarks, and the stages
that bridge the session into the graph (Q&A, traces, distilled lessons)
append it to their own node set (``bridge_node_set``), so a
``node_name``-scoped recall sees what the session stored.

Every input path goes through ``normalize_node_set``, so the entry field, the
call-level kwarg and the pin accept and refuse exactly the same values.

The rows are read and written through the session manager's context-entry
methods, the surface the improve watermarks use. Those methods fail open on
infrastructure errors; here neither direction may. A pin that cannot be
written is refused instead of being silently dropped, and every read asks the
manager to re-raise, so an unreadable session is never mistaken for an
unpinned one: that would let a conflicting set through at pin time, and let
the bridging stages add a pinned session's text to the graph untagged. When
pinning, either failure raises ``SessionNodeSetUnavailableError`` (HTTP 503),
so a cache outage is never reported as a conflict.

Lifetime: the pin row expires with the session's other context rows. Writing
Q&A or traces refreshes only their own rows, so a matching repeat of the pin
rewrites the pin row to keep its expiry moving with tagged traffic.

Concurrency: the pin's read-check-write runs under an in-process
``asyncio.Lock`` of its own (``session_lock``), not the session turn lock, so a
tagged write never waits behind a recall's LLM answer. It makes the
one-set-per-session guarantee hold within one worker process. Across processes
there is no lock today: two workers taking the first write of the same session
at the same moment can each see no pin and each write one. Readers take the
last pin row, which is the last write on every cache backend (SQL upserts the
row; Redis and the filesystem cache append).
"""

from collections.abc import Iterable

from fastapi import status

from cognee.exceptions import CogneeApiError, CogneeTransientError
from cognee.infrastructure.locks.session_lock import session_lock

SESSION_NODE_SET_STATE_ID = "session_node_set"
SESSION_NODE_SET_STATE_KIND = "session_node_set_state"


class SessionNodeSetConflictError(CogneeApiError):
    """The session already carries a different ``node_set``."""

    def __init__(self, pinned: tuple[str, ...], requested: list[str]):
        super().__init__(
            message=(
                "A session's node_set cannot change once pinned "
                f"(pinned: {sorted(pinned)}, requested: {requested}); start a new session"
            ),
            name="SessionNodeSetConflictError",
            status_code=status.HTTP_409_CONFLICT,
            log=False,
        )


class SessionNodeSetUnavailableError(CogneeTransientError):
    """The session cache could not be read or written while pinning ``node_set``."""

    def __init__(self, message: str):
        super().__init__(message=message, name="SessionNodeSetUnavailableError")


def normalize_node_set(value) -> list[str] | None:
    """The one validation and normalization rule for a session ``node_set``.

    ``None`` stays ``None`` (no node set given). Anything else must be a list or
    tuple of non-blank strings; the result is deduplicated and sorted, so
    equality is order-independent and duplicates collapse. An empty list stays
    an empty list, which pins nothing. A bare string is refused rather than
    split into one-letter names, and a blank name is refused because it would
    pin a node set with no name.

    Raises ``ValueError`` (which Pydantic reports as a validation error, and
    the HTTP API as a 400) for every invalid value.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "node_set must be a list of node-set names, not a single string; "
            f"use [{value!r}] for one name"
        )
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"node_set must be a list of node-set names, got {type(value).__name__}")
    for name in value:
        if not isinstance(name, str):
            raise ValueError(f"node_set names must be strings, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("node_set names must not be empty")
    return sorted(set(value))


def bridge_node_set(stage_node_sets: Iterable[str], pinned: Iterable[str]) -> list[str]:
    """The node set a bridged item is added under: the stage's own sets, then the pin.

    One rule for every bridge (Q&A, traces, distilled lessons): stage sets keep
    their order and come first, the session's pinned set follows, duplicates
    are dropped. Dropping a stage set here would change what a stage-scoped
    recall finds, so callers pass all of theirs.
    """
    return list(dict.fromkeys([*stage_node_sets, *pinned]))


def node_set_from_rows(rows: Iterable | None) -> tuple[str, ...]:
    """Parse the pinned ``node_set`` out of already-loaded session-context rows.

    Takes the last pin row: Redis and the filesystem cache append instead of
    replacing, so after a cross-process race the last row is the last write.
    """
    pins = [
        row
        for row in rows or []
        if isinstance(row, dict) and row.get("id") == SESSION_NODE_SET_STATE_ID
    ]
    row = pins[-1] if pins else None
    return tuple(sorted(row.get("node_set") or [])) if row else ()


async def read_session_context_strict(session_manager, user_id: str, session_id: str) -> list:
    """All session-context rows, raising if the cache cannot be read.

    Readers that need the pin and something else stored beside it (the
    bridging extractors read the persist watermark too) take one snapshot here
    and parse both out of it.
    """
    return await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id, raise_on_error=True
    )


async def get_session_node_set(session_manager, user_id: str, session_id: str) -> tuple[str, ...]:
    """The session's pinned ``node_set``, sorted; empty when none was pinned.

    Raises when the session cache cannot be read: callers decide whether that
    refuses a pin or skips a session for this run, never treat it as "no pin".
    """
    return node_set_from_rows(
        await read_session_context_strict(session_manager, user_id, session_id)
    )


async def pin_session_node_set(session_manager, user_id: str, session_id: str, node_set) -> None:
    """Pin ``node_set`` on the session, or verify it equals the set already pinned.

    ``node_set`` goes through ``normalize_node_set`` first. An empty list pins
    nothing: it means the same as an entry without ``node_set``, so a client
    that sends ``node_set: []`` on its first turn can still pin one later. Runs
    under a per-session lock so two first entries in this process cannot both
    pin (see the module docstring for the cross-process gap).

    Raises ``SessionNodeSetConflictError`` (409) for a different pinned set and
    ``SessionNodeSetUnavailableError`` (503) when the cache cannot be read or
    rejects the pin.
    """
    requested = normalize_node_set(node_set)
    if not requested:
        return
    async with session_lock(f"{user_id}:{session_id}", "node_set"):
        try:
            pinned = await get_session_node_set(session_manager, user_id, session_id)
        except CogneeApiError:
            raise
        except Exception as error:
            raise SessionNodeSetUnavailableError(
                f"Could not read the node_set pinned on session {session_id}: "
                "the session cache is unavailable"
            ) from error
        if pinned:
            if list(pinned) != requested:
                raise SessionNodeSetConflictError(pinned, requested)
            # Rewrite the pin so its expiry slides with tagged traffic; Q&A and
            # trace writes refresh only their own rows. Best effort: the pin is
            # still there, so a failed refresh must not fail the write.
            await session_manager.update_session_context_entry(
                user_id=user_id,
                session_id=session_id,
                entry_id=SESSION_NODE_SET_STATE_ID,
                merge={"node_set": requested},
            )
            return
        written = await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump={
                "id": SESSION_NODE_SET_STATE_ID,
                "kind": SESSION_NODE_SET_STATE_KIND,
                "node_set": requested,
            },
        )
        if not written:
            raise SessionNodeSetUnavailableError(
                f"Could not pin node_set {requested} on session {session_id}: "
                "the session cache rejected the write"
            )
