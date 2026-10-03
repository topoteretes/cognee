"""forget(everything=True) must only clear the caller's session cache.

The session cache is shared by every user. Pruning it wholesale (Redis
FLUSHDB, diskcache clear) let any single user wipe every other user's
sessions, so the cleanup is scoped to the caller's own sessions.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

forget_module = importlib.import_module("cognee.api.v1.forget.forget")
datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
invalidate_module = importlib.import_module("cognee.modules.session_lifecycle.invalidate_sessions")
cache_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.cache.get_cache_engine"
)

USER = SimpleNamespace(id=uuid4())


@pytest.mark.asyncio
async def test_forget_everything_invalidates_only_callers_sessions():
    cache_engine = MagicMock()
    cache_engine.prune = AsyncMock()
    invalidate = AsyncMock(return_value={"sessions_considered": 2, "sessions_deleted": 2})

    with (
        patch.object(
            datasets_module.datasets,
            "list_datasets",
            new=AsyncMock(return_value=[SimpleNamespace(id=uuid4())]),
        ),
        patch.object(datasets_module.datasets, "delete_all", new=AsyncMock()),
        patch.object(invalidate_module, "invalidate_sessions_for_user", new=invalidate),
        patch.object(cache_engine_module, "get_cache_engine", return_value=cache_engine),
    ):
        result = await forget_module._forget_everything(USER)

    assert result == {"datasets_removed": 1, "status": "success"}
    invalidate.assert_awaited_once_with(USER.id)
    cache_engine.prune.assert_not_awaited()


@pytest.mark.asyncio
async def test_forget_everything_session_cleanup_failure_is_nonfatal():
    with (
        patch.object(datasets_module.datasets, "list_datasets", new=AsyncMock(return_value=[])),
        patch.object(datasets_module.datasets, "delete_all", new=AsyncMock()),
        patch.object(
            invalidate_module,
            "invalidate_sessions_for_user",
            new=AsyncMock(side_effect=RuntimeError("cache down")),
        ),
    ):
        result = await forget_module._forget_everything(USER)

    assert result == {"datasets_removed": 0, "status": "success"}
