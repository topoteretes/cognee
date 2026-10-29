"""Sessions that cited a retired decision are invalidated with it (SDK-1328)."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

methods = importlib.import_module("cognee.modules.conflicts.methods")


@pytest.mark.asyncio
async def test_retired_decisions_reach_session_invalidation():
    dataset, user = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    invalidate = AsyncMock()
    with (
        patch.object(methods, "retire_forgotten_decisions", AsyncMock(return_value={"c1"})),
        patch.object(methods, "invalidate_sessions_for_deleted_data", invalidate),
    ):
        assert await methods.retire_decisions_after_edit(dataset, user) == {"c1"}

    invalidate.assert_awaited_once_with(dataset.id, {"c1"}, set(), user_id=user.id)


@pytest.mark.asyncio
async def test_a_failed_session_invalidation_fails_the_retirement():
    """Cached answers that cite a retired decision must not survive it silently."""
    with (
        patch.object(methods, "retire_forgotten_decisions", AsyncMock(return_value={"c1"})),
        patch.object(
            methods,
            "invalidate_sessions_for_deleted_data",
            AsyncMock(side_effect=RuntimeError("cache down")),
        ),
        pytest.raises(RuntimeError, match="cache down"),
    ):
        await methods.retire_decisions_after_edit(
            SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
        )


@pytest.mark.asyncio
async def test_nothing_retired_invalidates_nothing():
    invalidate = AsyncMock()
    with (
        patch.object(methods, "retire_forgotten_decisions", AsyncMock(return_value=set())),
        patch.object(methods, "invalidate_sessions_for_deleted_data", invalidate),
    ):
        await methods.retire_decisions_after_edit(
            SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
        )

    invalidate.assert_not_awaited()
