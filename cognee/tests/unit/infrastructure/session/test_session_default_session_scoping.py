"""Dataset-scoped default session IDs.

When a caller omits session_id, the SessionManager derives the default from
the dataset it is scoped to (``default_session_<dataset_id>``) instead of the
single global ``"default_session"`` — so omitting session_id in two different
datasets can never mix their turns in one session. Explicit session IDs are
stored unchanged.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.session.session_manager import SessionManager


class TestDatasetBinding:
    """Default-session derivation from the manager's dataset binding."""

    @pytest.mark.asyncio
    async def test_explicit_dataset_derives_default_session_id(self):
        dataset_id = uuid4()
        manager = SessionManager(cache_engine=None, dataset_id=dataset_id)
        assert await manager._resolve_session_id(None) == f"default_session_{dataset_id}"

    @pytest.mark.asyncio
    async def test_no_dataset_and_no_default_uses_plain_default(self):
        """Without a resolvable main_dataset there is nothing to scope to."""
        manager = SessionManager(cache_engine=None)
        with patch.object(SessionManager, "_effective_dataset_id", AsyncMock(return_value=None)):
            assert await manager._resolve_session_id(None) == "default_session"

    @pytest.mark.asyncio
    async def test_explicit_session_id_unchanged(self):
        manager = SessionManager(cache_engine=None, dataset_id=uuid4())
        assert await manager._resolve_session_id("my_session") == "my_session"

    @pytest.mark.asyncio
    async def test_inherits_current_dataset_id_context(self):
        dataset_id = uuid4()
        token = current_dataset_id.set(dataset_id)
        try:
            manager = SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == dataset_id
        assert await manager._resolve_session_id(None) == f"default_session_{dataset_id}"

    def test_explicit_dataset_overrides_context(self):
        explicit_id = uuid4()
        token = current_dataset_id.set(uuid4())
        try:
            manager = SessionManager(cache_engine=None, dataset_id=explicit_id)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == explicit_id

    def test_non_uuid_dataset_context_raises(self):
        """A name cannot scope a session — constructing against one must break,
        not silently degrade to an unscoped session. (The database context
        manager resolves names before publishing, so this only fires on direct
        contextvar writes or bad constructor arguments.)"""
        from cognee.infrastructure.databases.exceptions import SessionParameterValidationError

        token = current_dataset_id.set("main_dataset")
        try:
            with pytest.raises(SessionParameterValidationError, match="main_dataset"):
                SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)


class TestDefaultDatasetFallback:
    """An omitted dataset falls back to main_dataset, like everywhere else in Cognee."""

    @pytest.mark.asyncio
    async def test_falls_back_to_main_dataset(self):
        user_id, dataset_id = uuid4(), uuid4()
        dataset = MagicMock()
        dataset.id = dataset_id
        manager = SessionManager(cache_engine=None)

        with patch(
            "cognee.modules.data.methods.get_datasets_by_name",
            AsyncMock(return_value=[dataset]),
        ) as by_name:
            resolved = await manager._resolve_session_id(None, str(user_id))
            # Second call must not re-query — the lookup is memoised per instance.
            await manager._resolve_session_id(None, str(user_id))

        assert resolved == f"default_session_{dataset_id}"
        by_name.assert_awaited_once_with(["main_dataset"], user_id)

    @pytest.mark.asyncio
    async def test_missing_main_dataset_does_not_create_one(self):
        """Reads must stay side-effect free before any dataset exists."""
        manager = SessionManager(cache_engine=None)
        with patch(
            "cognee.modules.data.methods.get_datasets_by_name",
            AsyncMock(return_value=[]),
        ):
            assert await manager._resolve_session_id(None, str(uuid4())) == "default_session"

    @pytest.mark.asyncio
    async def test_explicit_dataset_skips_the_fallback(self):
        manager = SessionManager(cache_engine=None, dataset_id=uuid4())
        with patch(
            "cognee.modules.data.methods.get_datasets_by_name",
            AsyncMock(return_value=[]),
        ) as by_name:
            await manager._resolve_session_id(None, str(uuid4()))
        by_name.assert_not_awaited()
