"""Regression tests for the API-key auth pool-deadlock fix (issue #4197).

`UserManager.get_by_token` used to look up the API key in one session and then
call `self.get()` — which borrows the request-scoped `get_user_db` session —
*while its own session was still open*. Every concurrent authenticated request
therefore pinned two pooled connections at once; once concurrency reached the
pool size this became a circular wait that deadlocked the pool and stranded
backends "idle in transaction", taking auth (and the whole API) down.

The fix resolves both the API key and the user inside a single short-lived
session and never calls `self.get()`, so auth holds exactly one connection and
releases it immediately. These tests lock in that invariant without needing a
real database.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.users.get_user_manager import UserManager

MODULE = "cognee.modules.users.get_user_manager"


class _FakeSession:
    """Async-context-manager session whose execute() returns queued results."""

    def __init__(self, results):
        self._results = list(results)
        self.execute = AsyncMock(side_effect=self._results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _result(scalar_value):
    res = MagicMock()
    res.scalar.return_value = scalar_value
    return res


def _engine_yielding(session):
    """Fake relational engine whose get_async_session() yields `session` once."""
    engine = MagicMock()
    engine.get_async_session = MagicMock(return_value=session)
    return engine


def _manager_with_get_spy():
    manager = UserManager(MagicMock())
    manager.get = AsyncMock(side_effect=AssertionError("self.get() must not be called"))
    return manager


@pytest.mark.asyncio
async def test_valid_key_resolves_user_in_one_session_without_self_get():
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, is_active=True)
    session = _FakeSession([_result(SimpleNamespace(user_id=user_id)), _result(user)])
    engine = _engine_yielding(session)
    manager = _manager_with_get_spy()

    with patch(f"{MODULE}.get_relational_engine", return_value=engine):
        result = await manager.get_by_token("raw-key")

    assert result is user
    # Exactly one session opened (single connection), both queries run on it.
    engine.get_async_session.assert_called_once()
    assert session.execute.await_count == 2
    # self.get() (the borrowed second connection) must never be used.
    manager.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_key_returns_none_without_user_lookup():
    session = _FakeSession([_result(None)])
    engine = _engine_yielding(session)
    manager = _manager_with_get_spy()

    with patch(f"{MODULE}.get_relational_engine", return_value=engine):
        result = await manager.get_by_token("bad-key")

    assert result is None
    engine.get_async_session.assert_called_once()
    # Only the API-key lookup runs; no user query, no self.get().
    assert session.execute.await_count == 1
    manager.get.assert_not_awaited()
