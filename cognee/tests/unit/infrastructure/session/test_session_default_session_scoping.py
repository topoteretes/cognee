"""Dataset-scoped default session IDs.

When a caller omits session_id and the SessionManager knows its dataset
(constructor argument or the dataset context variable), the default resolves
to ``default_session_<dataset_id>`` instead of the single global
``"default_session"`` — so omitting session_id in two different datasets can
never mix their turns in one session. Without a known dataset the plain
global default is used, matching the previous behavior. Explicit session IDs
are stored unchanged.
"""

from unittest.mock import patch
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.session.session_manager import SessionManager


class TestDatasetScopedDefaults:
    def test_explicit_dataset_derives_default_session_id(self):
        dataset_id = uuid4()
        manager = SessionManager(cache_engine=None, dataset_id=dataset_id)
        assert manager.resolve_session_id(None) == f"default_session_{dataset_id}"

    def test_no_dataset_uses_plain_default(self):
        """Without a known dataset there is nothing to scope to."""
        manager = SessionManager(cache_engine=None)
        assert manager.resolve_session_id(None) == "default_session"

    def test_explicit_session_id_unchanged(self):
        manager = SessionManager(cache_engine=None, dataset_id=uuid4())
        assert manager.resolve_session_id("my_session") == "my_session"

    def test_inherits_current_dataset_id_context(self):
        dataset_id = uuid4()
        token = current_dataset_id.set(dataset_id)
        try:
            manager = SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == dataset_id
        assert manager.resolve_session_id(None) == f"default_session_{dataset_id}"

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
        manager only accepts UUIDs, so this only fires on direct contextvar
        writes or bad constructor arguments.)"""
        from cognee.infrastructure.databases.exceptions import SessionParameterValidationError

        token = current_dataset_id.set("main_dataset")
        try:
            with pytest.raises(SessionParameterValidationError, match="main_dataset"):
                SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)

    def test_resolution_does_not_touch_the_database(self):
        """Resolution is pure: dataset known or not, no lookups happen."""
        with patch(
            "cognee.infrastructure.databases.relational.get_relational_engine"
        ) as engine_mock:
            SessionManager(cache_engine=None).resolve_session_id(None)
            SessionManager(cache_engine=None, dataset_id=uuid4()).resolve_session_id(None)
        engine_mock.assert_not_called()
