"""Tests for the per-session lock primitives, including cross-event-loop use.

The cross-loop test exercises the public API from more than one event loop.
Before the module-level locks were made loop-bound, the second loop raised
``RuntimeError: ... is bound to a different event loop`` once a lock was
contended. That failure is contention-gated on CPython 3.12+ (asyncio's
uncontended acquire fast-path skips the loop-binding check), so the test forces
real contention -- two concurrent operations on the same key -- otherwise it
would pass even against the buggy code.

The remaining tests pin the locking semantics the fix must preserve: same-key
operations serialize, different-key operations run concurrently.
"""

import asyncio

from cognee.infrastructure.locks.session_lock import session_lock


def _run_contended_same_key():
    async def scenario():
        async def op():
            async with session_lock("session-1", "update_qa"):
                await asyncio.sleep(0.01)

        # Two concurrent ops on the same key: one holds the per-key lock across
        # an await while the other waits, forcing the contended acquire path.
        await asyncio.gather(op(), op())

    asyncio.run(scenario())


def test_session_lock_works_across_event_loops():
    # Loop #1 binds the module-level locks to itself; loop #2 (a second
    # asyncio.run) must not raise "bound to a different event loop".
    _run_contended_same_key()
    _run_contended_same_key()


def test_session_lock_serializes_same_key_within_a_loop():
    async def scenario():
        events: list[tuple[str, int]] = []

        async def op(n: int):
            async with session_lock("session-1", "write"):
                events.append(("enter", n))
                await asyncio.sleep(0.01)
                events.append(("exit", n))

        await asyncio.gather(op(1), op(2))
        return events

    events = asyncio.run(scenario())

    # No interleaving: each enter is immediately followed by its own exit.
    assert [e[0] for e in events] == ["enter", "exit", "enter", "exit"]


def test_session_lock_allows_concurrency_across_different_keys():
    async def scenario():
        events: list[tuple[str, str]] = []

        async def op(key: str):
            async with session_lock(key, "write"):
                events.append(("enter", key))
                await asyncio.sleep(0.02)
                events.append(("exit", key))

        await asyncio.gather(op("a"), op("b"))
        return events

    events = asyncio.run(scenario())

    # Different keys do not block each other: both enter before either exits.
    assert [e[0] for e in events[:2]] == ["enter", "enter"]
