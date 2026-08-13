import asyncio

import pytest

from cognee.infrastructure.locks import session_turn_lock


@pytest.mark.asyncio
async def test_session_turn_lock_serializes_same_user_and_session():
    active = 0
    peak = 0

    async def run_turn():
        nonlocal active, peak
        async with session_turn_lock("u1", "s1"):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(run_turn(), run_turn())

    assert peak == 1


@pytest.mark.asyncio
async def test_session_turn_lock_does_not_mix_users():
    both_entered = asyncio.Event()
    active = 0

    async def run_turn(user_id):
        nonlocal active
        async with session_turn_lock(user_id, "s1"):
            active += 1
            if active == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)

    await asyncio.gather(run_turn("u1"), run_turn("u2"))


def test_session_turn_lock_can_be_reused_across_event_loops():
    async def run_once():
        async with session_turn_lock("u1", "s1"):
            return True

    assert asyncio.run(run_once()) is True
    assert asyncio.run(run_once()) is True
