"""The improve lock's rerun request (SDK-593): a lock loser asks the holder for one more pass.

In-process registry semantics only — the same scope as the claim itself.
"""

import importlib

import pytest

session_lock = importlib.import_module("cognee.infrastructure.locks.session_lock")


@pytest.fixture(autouse=True)
def _clean_registry():
    session_lock._improving_sessions.clear()
    session_lock._rerun_requested.clear()
    yield
    session_lock._improving_sessions.clear()
    session_lock._rerun_requested.clear()


@pytest.mark.asyncio
async def test_request_lands_only_on_held_keys():
    assert await session_lock.try_acquire_improve_lock_many(["session:u:a", "dataset:d"])

    assert await session_lock.request_improve_rerun_many(["session:u:a"]) is True
    # A free key has nobody to fulfil the request: not recorded, not "busy".
    assert await session_lock.request_improve_rerun_many(["session:u:free"]) is False
    assert session_lock._rerun_requested == {"session:u:a"}


@pytest.mark.asyncio
async def test_release_or_rerun_consumes_a_pending_request_and_keeps_every_key():
    keys = ["session:u:a", "dataset:d"]
    assert await session_lock.try_acquire_improve_lock_many(keys)
    await session_lock.request_improve_rerun_many(["session:u:a"])

    kept = await session_lock.release_or_rerun_improve_lock_many(keys, rerun_keys=["session:u:a"])

    assert kept is False  # "run once more"
    assert not await session_lock.try_acquire_improve_lock_many(["dataset:d"])  # still held
    assert session_lock._rerun_requested == set()  # consumed exactly once

    released = await session_lock.release_or_rerun_improve_lock_many(
        keys, rerun_keys=["session:u:a"]
    )
    assert released is True
    assert await session_lock.try_acquire_improve_lock_many(keys)


@pytest.mark.asyncio
async def test_request_on_the_dataset_key_alone_is_ignored_by_a_session_scoped_release():
    """The protocol is session-only: a dataset-key request never triggers a rerun."""
    keys = ["session:u:a", "dataset:d"]
    assert await session_lock.try_acquire_improve_lock_many(keys)
    await session_lock.request_improve_rerun_many(["dataset:d"])

    assert (
        await session_lock.release_or_rerun_improve_lock_many(keys, rerun_keys=["session:u:a"])
        is True
    )
    assert await session_lock.try_acquire_improve_lock_many(keys)


@pytest.mark.asyncio
async def test_a_fresh_claim_clears_stale_requests():
    """A new holder starts with a full watermark pass, which is what the request asked for."""
    assert await session_lock.try_acquire_improve_lock_many(["session:u:a"])
    await session_lock.request_improve_rerun_many(["session:u:a"])
    # Force release (error path / pass bound) leaves the request behind...
    await session_lock.release_improve_lock_many(["session:u:a"])
    assert "session:u:a" in session_lock._rerun_requested

    # ...and the next claim clears it.
    assert await session_lock.try_acquire_improve_lock_many(["session:u:a"])
    assert "session:u:a" not in session_lock._rerun_requested


@pytest.mark.asyncio
async def test_empty_keys_are_noops():
    assert await session_lock.request_improve_rerun_many([]) is False
    assert await session_lock.request_improve_rerun_many(["", None]) is False
    assert await session_lock.release_or_rerun_improve_lock_many([], rerun_keys=[]) is True
