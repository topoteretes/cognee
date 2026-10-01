"""Tests for the shared telemetry aiohttp session lifecycle.

The session is a process-wide singleton reused across telemetry calls. It used
to leak: nothing closed it at interpreter exit, and a loop change dropped the
old session without closing it — both printed "Unclosed client session" from
aiohttp's destructor in SDK scripts.
"""

import asyncio

import pytest

from cognee.shared import utils as shared_utils


@pytest.fixture(autouse=True)
def _reset_session_state():
    yield
    # Never leave a session behind for other tests (or the test runner's exit).
    session = shared_utils._telemetry_session
    if session is not None and not session.closed:
        asyncio.run(session.close())
    shared_utils._telemetry_session = None
    shared_utils._telemetry_session_loop = None


@pytest.mark.asyncio
async def test_close_telemetry_session_closes_and_resets():
    session = await shared_utils._get_telemetry_session()
    assert not session.closed

    await shared_utils.close_telemetry_session()

    assert session.closed
    assert shared_utils._telemetry_session is None
    assert shared_utils._telemetry_session_loop is None

    # Repeated close is a no-op, and the getter rebuilds transparently after.
    await shared_utils.close_telemetry_session()
    rebuilt = await shared_utils._get_telemetry_session()
    assert not rebuilt.closed and rebuilt is not session
    await shared_utils.close_telemetry_session()


def test_loop_change_closes_stale_session_instead_of_dropping_it():
    async def get_session():
        return await shared_utils._get_telemetry_session()

    first = asyncio.run(get_session())
    assert not first.closed

    # New asyncio.run = new loop: the getter must rebuild AND close the stale
    # session (aiohttp accepts close() from a different loop).
    second = asyncio.run(get_session())

    assert second is not first
    assert first.closed
    assert not second.closed


@pytest.mark.asyncio
async def test_atexit_hook_registered_once(monkeypatch):
    registered = []
    import atexit

    monkeypatch.setattr(shared_utils, "_telemetry_atexit_registered", False)
    monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn))

    await shared_utils._get_telemetry_session()
    await shared_utils.close_telemetry_session()
    await shared_utils._get_telemetry_session()

    assert len(registered) == 1
