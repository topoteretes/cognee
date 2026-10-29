"""Sessions that cited a retired decision are invalidated, best-effort (SDK-1328)."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

methods = importlib.import_module("cognee.modules.conflicts.methods")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation_fails", [False, True])
async def test_retired_decisions_reach_session_invalidation(invalidation_fails):
    dataset, user = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    invalidate = AsyncMock(side_effect=RuntimeError("cache down") if invalidation_fails else None)
    with (
        patch.object(methods, "retire_forgotten_decisions", AsyncMock(return_value={"c1"})),
        patch.object(methods, "invalidate_sessions_for_deleted_data", invalidate),
    ):
        assert await methods.retire_decisions_after_rebuild(dataset, user) == {"c1"}

    invalidate.assert_awaited_once_with(dataset.id, {"c1"}, set(), user_id=user.id)


@pytest.mark.asyncio
async def test_nothing_retired_invalidates_nothing():
    invalidate = AsyncMock()
    with (
        patch.object(methods, "retire_forgotten_decisions", AsyncMock(return_value=set())),
        patch.object(methods, "invalidate_sessions_for_deleted_data", invalidate),
    ):
        await methods.retire_decisions_after_rebuild(
            SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
        )

    invalidate.assert_not_awaited()
