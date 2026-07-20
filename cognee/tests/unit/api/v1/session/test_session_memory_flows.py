import asyncio
import importlib
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from cognee.api.v1.session import SessionQAEntry
from cognee.modules.recall.types.RecallResponse import ResponseQAEntry
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType

# Import actual modules via importlib to avoid __init__.py name shadowing.
# Several __init__.py files re-export functions with the same name as their
# module (e.g., `from .get_session_manager import get_session_manager`),
# which causes mock.patch()'s getattr-based resolution to find the function
# instead of the submodule on Python ≤3.12.
_mod_sm = importlib.import_module("cognee.infrastructure.session.get_session_manager")
_pkg_improve = importlib.import_module("cognee.api.v1.improve")
_mod_query_router = importlib.import_module("cognee.api.v1.recall.query_router")
_mod_search_methods = importlib.import_module("cognee.modules.search.methods.search")


@asynccontextmanager
async def _noop_database_context(*_args, **_kwargs):
    yield


@contextmanager
def _patch_remember_startup():
    with (
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new=AsyncMock()),
        patch("cognee.modules.engine.operations.setup.setup", new=AsyncMock()),
        patch(
            "cognee.context_global_variables.set_database_global_context_variables",
            new=_noop_database_context,
        ),
    ):
        yield


def _get_remember_module():
    return importlib.import_module("cognee.api.v1.remember.remember")


def _get_improve_module():
    return importlib.import_module("cognee.api.v1.improve")


def _get_query_router_module():
    return importlib.import_module("cognee.api.v1.recall.query_router")


# ---------------------------------------------------------------------------
# remember() passes session_ids to improve()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_passes_session_ids_to_improve():
    """When self_improvement=True and session_ids given, improve() receives them."""
    improve_calls = []

    async def mock_improve(**kwargs):
        improve_calls.append(kwargs)

    mock_user = MagicMock()
    mock_user.id = "u1"

    with (
        _patch_remember_startup(),
        patch("cognee.api.v1.add.add", AsyncMock()),
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={"status": "ok"})),
        patch.object(_pkg_improve, "improve", mock_improve),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(return_value=mock_user),
        ),
        patch.object(
            _get_remember_module(),
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(mock_user, "")),
        ),
    ):
        from cognee.api.v1.remember.remember import remember

        await remember(
            "test data",
            dataset_name="test_ds",
            self_improvement=True,
            session_ids=["s1", "s2"],
        )

    assert len(improve_calls) == 1
    assert improve_calls[0]["session_ids"] == ["s1", "s2"]
    assert improve_calls[0]["dataset"] == "test_ds"


@pytest.mark.asyncio
async def test_remember_no_session_ids_skips_in_improve():
    """When session_ids not provided, improve() is called without them."""
    improve_calls = []

    async def mock_improve(**kwargs):
        improve_calls.append(kwargs)

    mock_user = MagicMock()
    mock_user.id = "u1"

    with (
        _patch_remember_startup(),
        patch("cognee.api.v1.add.add", AsyncMock()),
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={"status": "ok"})),
        patch.object(_pkg_improve, "improve", mock_improve),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(return_value=mock_user),
        ),
        patch.object(
            _get_remember_module(),
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(mock_user, "")),
        ),
    ):
        from cognee.api.v1.remember.remember import remember

        await remember("test data", self_improvement=True)

    assert len(improve_calls) == 1
    assert "session_ids" not in improve_calls[0]


# ---------------------------------------------------------------------------
# RememberResult
# ---------------------------------------------------------------------------


class TestRememberResult:
    def test_repr_basic(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="test")
        assert "completed" in repr(r)
        assert "test" in repr(r)

    def test_repr_includes_elapsed(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="test")
        r.elapsed_seconds = 4.2
        assert "elapsed=4.2s" in repr(r)

    def test_repr_includes_error(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="errored", dataset_name="test")
        r.error = "something broke"
        assert "something broke" in repr(r)

    def test_bool_completed_is_true(self):
        from cognee.api.v1.remember.remember import RememberResult

        assert bool(RememberResult(status="completed", dataset_name="x"))
        assert bool(RememberResult(status="session_stored", dataset_name="x"))

    def test_bool_running_is_false(self):
        from cognee.api.v1.remember.remember import RememberResult

        assert not bool(RememberResult(status="running", dataset_name="x"))
        assert not bool(RememberResult(status="errored", dataset_name="x"))

    def test_done_without_task(self):
        from cognee.api.v1.remember.remember import RememberResult

        assert RememberResult(status="completed", dataset_name="x").done is True
        assert RememberResult(status="errored", dataset_name="x").done is True
        assert RememberResult(status="running", dataset_name="x").done is False

    def test_done_with_task(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="x")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        r._task = mock_task
        assert r.done is False

        mock_task.done.return_value = True
        assert r.done is True

    @pytest.mark.asyncio
    async def test_await_completed_returns_self(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x")
        result = await r
        assert result is r

    @pytest.mark.asyncio
    async def test_await_background_task(self):
        """Awaiting a result with a background task waits for the task."""
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="x")
        completed = False

        async def background():
            nonlocal completed
            await asyncio.sleep(0.01)
            r.status = "completed"
            completed = True

        r._task = asyncio.create_task(background())
        assert not completed
        result = await r
        assert result is r
        assert completed
        assert r.status == "completed"

    def test_resolve_extracts_pipeline_info(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="test")

        mock_run_info = MagicMock()
        mock_run_info.status = "PipelineRunCompleted"
        mock_run_info.pipeline_run_id = UUID("12345678-1234-1234-1234-123456789abc")

        ds_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        r._resolve({ds_id: mock_run_info})

        assert r.status == "completed"
        assert r.dataset_id == str(ds_id)
        assert r.pipeline_run_id == str(mock_run_info.pipeline_run_id)
        assert r.raw_result is not None
        assert r.elapsed_seconds is not None
        assert r.elapsed_seconds >= 0

    def test_resolve_extracts_item_info(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="test")

        mock_data_item = MagicMock()
        mock_data_item.id = UUID("11111111-2222-3333-4444-555555555555")
        mock_data_item.name = "einstein.txt"
        mock_data_item.content_hash = "abc123"
        mock_data_item.token_count = 42
        mock_data_item.mime_type = "text/plain"
        mock_data_item.data_size = 1024

        mock_run_info = MagicMock()
        mock_run_info.status = "PipelineRunCompleted"
        mock_run_info.pipeline_run_id = UUID("12345678-1234-1234-1234-123456789abc")
        mock_run_info.payload = [mock_data_item]

        ds_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        r._resolve({ds_id: mock_run_info})

        assert r.items_processed == 1
        assert r.content_hash == "abc123"
        assert r.items[0]["name"] == "einstein.txt"
        assert r.items[0]["token_count"] == 42
        assert r.items[0]["mime_type"] == "text/plain"
        assert r.items[0]["data_size"] == 1024
        assert "items=1" in repr(r)
        assert "abc123" in repr(r)

    def test_resolve_detects_error(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="test")

        mock_run_info = MagicMock()
        mock_run_info.status = "PipelineRunErrored"

        r._resolve({UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"): mock_run_info})
        assert r.status == "errored"

    def test_fail_sets_error_and_elapsed(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="test")
        r._fail(ValueError("test error"))
        assert r.status == "errored"
        assert r.error == "test error"
        assert r.elapsed_seconds is not None

    @pytest.mark.asyncio
    async def test_remember_returns_remember_result(self):
        """Blocking remember() returns a properly resolved RememberResult."""
        from cognee.api.v1.remember.remember import RememberResult

        mock_user = MagicMock()
        mock_user.id = "u1"

        with (
            _patch_remember_startup(),
            patch("cognee.api.v1.add.add", AsyncMock()),
            patch(
                "cognee.api.v1.cognify.cognify",
                AsyncMock(return_value={}),
            ),
            patch.object(_pkg_improve, "improve", AsyncMock()),
            patch(
                "cognee.modules.users.methods.get_default_user",
                AsyncMock(return_value=mock_user),
            ),
            patch.object(
                _get_remember_module(),
                "resolve_authorized_user_datasets",
                AsyncMock(return_value=(mock_user, "")),
            ),
        ):
            from cognee.api.v1.remember.remember import remember

            result = await remember("test data")

        assert isinstance(result, RememberResult)
        assert result.status == "completed"
        assert result.dataset_name == "main_dataset"
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds >= 0

    @pytest.mark.asyncio
    async def test_session_result_has_elapsed(self):
        """Raw session QA writes run in the selected dataset owner's DB context."""
        mock_user = MagicMock()
        mock_user.id = "u1"

        mock_sm = MagicMock()
        mock_sm.is_available = True
        context_active = False

        async def add_qa(**_kwargs):
            assert context_active is True

        mock_sm.add_qa = AsyncMock(side_effect=add_qa)
        dataset = SimpleNamespace(id=uuid4(), name="main_dataset", owner_id=uuid4())
        context_calls = []

        @asynccontextmanager
        async def database_context(dataset_id, owner_id):
            nonlocal context_active
            context_calls.append((dataset_id, owner_id))
            context_active = True
            try:
                yield
            finally:
                context_active = False

        with (
            _patch_remember_startup(),
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                new=database_context,
            ),
            patch(
                "cognee.modules.users.methods.get_default_user",
                AsyncMock(return_value=mock_user),
            ),
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ) as get_session_manager,
            patch.object(
                _get_remember_module(),
                "resolve_authorized_user_datasets",
                AsyncMock(return_value=(mock_user, [dataset])),
            ),
        ):
            from cognee.api.v1.remember.remember import RememberResult, remember

            result = await remember("test data", session_id="s1", self_improvement=False)

        assert isinstance(result, RememberResult)
        assert result.status == "session_stored"
        assert result.session_id == "s1"
        assert result.session_ids == ["s1"]
        assert result.dataset_id == str(dataset.id)
        assert result.elapsed_seconds is not None
        get_session_manager.assert_called_once_with(dataset_id=dataset.id)
        assert context_calls == [(dataset.id, dataset.owner_id)]

    @pytest.mark.asyncio
    async def test_typed_session_entry_uses_canonical_dataset_scope(self):
        from cognee.api.v1.remember.remember import remember
        from cognee.memory import QAEntry

        user = SimpleNamespace(id=uuid4())
        dataset = SimpleNamespace(id=uuid4(), name="canonical", owner_id=uuid4())
        resolver = AsyncMock(return_value=(user, [dataset]))
        mock_sm = MagicMock(is_available=True)
        context_active = False

        async def add_qa(**_kwargs):
            assert context_active is True
            return "qa-1"

        mock_sm.add_qa = AsyncMock(side_effect=add_qa)
        context_calls = []

        @asynccontextmanager
        async def database_context(dataset_id, owner_id):
            nonlocal context_active
            context_calls.append((dataset_id, owner_id))
            context_active = True
            try:
                yield
            finally:
                context_active = False

        with (
            _patch_remember_startup(),
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                new=database_context,
            ),
            patch.object(
                _get_remember_module(),
                "resolve_authorized_user_datasets",
                resolver,
            ),
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ) as get_session_manager,
        ):
            result = await remember(
                QAEntry(question="Q", answer="A"),
                dataset_id=dataset.id,
                session_id="shared-id",
                user=user,
            )

        resolver.assert_awaited_once_with(dataset.id, user)
        get_session_manager.assert_called_once_with(dataset_id=dataset.id)
        assert result.dataset_name == "canonical"
        assert result.dataset_id == str(dataset.id)
        assert result.entry_id == "qa-1"
        assert context_calls == [(dataset.id, dataset.owner_id)]

    @pytest.mark.asyncio
    async def test_session_remember_does_not_write_when_dataset_is_unauthorized(self):
        from cognee.api.v1.remember.remember import remember
        from cognee.modules.users.exceptions import PermissionDeniedError

        user = SimpleNamespace(id=uuid4())
        get_session_manager = MagicMock()

        with (
            _patch_remember_startup(),
            patch.object(
                _get_remember_module(),
                "resolve_authorized_user_datasets",
                AsyncMock(side_effect=PermissionDeniedError()),
            ),
            patch.object(_mod_sm, "get_session_manager", get_session_manager),
        ):
            with pytest.raises(PermissionDeniedError):
                await remember(
                    "secret",
                    dataset_id=uuid4(),
                    session_id="shared-id",
                    user=user,
                    self_improvement=False,
                )

        get_session_manager.assert_not_called()


# ---------------------------------------------------------------------------
# RememberResult session_id / session_ids unification
# ---------------------------------------------------------------------------


class TestRememberResultSessions:
    def test_session_id_property_single(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x", session_ids=["s1"])
        assert r.session_id == "s1"
        assert r.session_ids == ["s1"]

    def test_session_id_property_multiple(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x", session_ids=["s1", "s2"])
        assert r.session_id is None
        assert r.session_ids == ["s1", "s2"]

    def test_session_id_property_none(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x")
        assert r.session_id is None
        assert r.session_ids is None

    def test_repr_single_session(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x", session_ids=["s1"])
        assert "session_id='s1'" in repr(r)
        assert "session_ids" not in repr(r)

    def test_repr_multiple_sessions(self):
        from cognee.api.v1.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x", session_ids=["s1", "s2"])
        assert "session_ids=" in repr(r)

    @pytest.mark.asyncio
    async def test_remember_permanent_carries_session_ids(self):
        """Permanent remember() with session_ids carries them in result."""
        mock_user = MagicMock()
        mock_user.id = "u1"

        with (
            _patch_remember_startup(),
            patch("cognee.api.v1.add.add", AsyncMock()),
            patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})),
            patch.object(_pkg_improve, "improve", AsyncMock()),
            patch(
                "cognee.modules.users.methods.get_default_user",
                AsyncMock(return_value=mock_user),
            ),
            patch.object(
                _get_remember_module(),
                "resolve_authorized_user_datasets",
                AsyncMock(return_value=(mock_user, "")),
            ),
        ):
            from cognee.api.v1.remember.remember import remember

            result = await remember("test data", session_ids=["s1", "s2"], self_improvement=True)

        assert result.session_ids == ["s1", "s2"]
        assert result.session_id is None  # multiple → None


# ---------------------------------------------------------------------------
# _search_session — word boundary matching
# ---------------------------------------------------------------------------


class TestSearchSession:
    @pytest.mark.asyncio
    async def test_word_boundary_matching(self):
        """'graph' should NOT match 'paragraph'."""
        from cognee.api.v1.recall.recall import _search_session

        mock_user = MagicMock()
        mock_user.id = "u1"

        entries = [
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question="What is a paragraph?",
                context="",
                answer="A block of text.",
            ),
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question="What is a graph?",
                context="",
                answer="Nodes and edges.",
            ),
        ]

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.get_session = AsyncMock(return_value=entries)

        with (
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ),
        ):
            results = await _search_session("graph", "s1", user=mock_user)

        assert len(results) == 1
        assert "graph" in results[0].question.lower()

    @pytest.mark.asyncio
    async def test_multiple_word_ranking(self):
        """Entries matching more query words rank higher."""
        from cognee.api.v1.recall.recall import _search_session

        mock_user = MagicMock()
        mock_user.id = "u1"

        entries = [
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question="Tell me about cats",
                context="",
                answer="Cats are animals.",
            ),
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question="Tell me about cats and dogs",
                context="",
                answer="Both are pets.",
            ),
        ]

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.get_session = AsyncMock(return_value=entries)

        with (
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ),
        ):
            results = await _search_session("cats dogs", "s1", user=mock_user)

        # Entry with both "cats" and "dogs" should rank first
        assert len(results) == 2
        assert "dogs" in results[0].question.lower()

    @pytest.mark.asyncio
    async def test_source_tagging(self):
        """Session results should have _source='session'."""
        from cognee.api.v1.recall.recall import _search_session

        mock_user = MagicMock()
        mock_user.id = "u1"

        entries = [
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question="What is Einstein?",
                context="",
                answer="A physicist.",
            ),
        ]

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.get_session = AsyncMock(return_value=entries)

        with (
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ),
        ):
            results = await _search_session("Einstein", "s1", user=mock_user)

        assert results[0].source == "session"

    @pytest.mark.asyncio
    async def test_empty_session(self):
        """Empty session returns empty list."""
        from cognee.api.v1.recall.recall import _search_session

        mock_user = MagicMock()
        mock_user.id = "u1"

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.get_session = AsyncMock(return_value=[])

        with (
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ),
        ):
            results = await _search_session("anything", "s1", user=mock_user)

        assert results == []

    @pytest.mark.asyncio
    async def test_short_words_ignored(self):
        """Single-character words like 'a' and 'I' are skipped."""
        from cognee.api.v1.recall.recall import _search_session

        mock_user = MagicMock()
        mock_user.id = "u1"

        entries = [
            SessionQAEntry(
                time=datetime.utcnow().isoformat(),
                question="I have a cat",
                context="",
                answer="Nice.",
            ),
        ]

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.get_session = AsyncMock(return_value=entries)

        with (
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ),
        ):
            # Query with only short words → no matches
            results = await _search_session("a I", "s1", user=mock_user)

        assert results == []


# ---------------------------------------------------------------------------
# recall() session-only vs graph fallthrough
# ---------------------------------------------------------------------------


def _get_recall_module():
    """Import the recall module by full path to avoid __init__.py name collision."""
    import importlib

    return importlib.import_module("cognee.api.v1.recall.recall")


class TestRecallSessionMode:
    @pytest.fixture(autouse=True)
    def _disable_telemetry(self, monkeypatch):
        monkeypatch.setattr("cognee.shared.utils.send_telemetry", lambda *args, **kwargs: None)

    @pytest.mark.asyncio
    async def test_session_only_when_no_datasets_no_type(self):
        """recall(session_id=X) without datasets/type searches session."""
        recall_mod = _get_recall_module()

        session_entries = [
            ResponseQAEntry(
                time=datetime.utcnow().isoformat(),
                question="test",
                context="",
                answer="result",
                source="session",
            )
        ]

        with patch.object(
            recall_mod,
            "_search_session",
            AsyncMock(return_value=session_entries),
        ):
            results = await recall_mod.recall("test", session_id="s1")

        assert len(results) == 1
        assert results[0].source == "session"

    @pytest.mark.asyncio
    async def test_fallthrough_to_graph_when_session_empty(self):
        """When session search returns nothing, falls through to graph."""
        recall_mod = _get_recall_module()

        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_payload = SearchResultPayload(
            result_object="graph result", search_type=SearchType.GRAPH_COMPLETION
        )

        with (
            patch.object(
                recall_mod,
                "_search_session",
                AsyncMock(return_value=[]),
            ),
            patch.object(
                _mod_search_methods,
                "authorized_search",
                AsyncMock(return_value=[mock_payload]),
            ),
            patch.object(
                _mod_query_router,
                "route_query",
                return_value=MagicMock(search_type=MagicMock()),
            ),
        ):
            results = await recall_mod.recall("test", session_id="s1", user=mock_user)

        # Should get graph results since session was empty
        assert len(results) == 1
        assert results[0].source == "graph"
        assert results[0].text == "graph result"

    @pytest.mark.asyncio
    async def test_explicit_query_type_skips_session_search(self):
        """When query_type is explicit, session search is skipped."""
        mock_payload = SearchResultPayload(
            result_object="graph result", search_type=SearchType.GRAPH_COMPLETION
        )
        mock_user = MagicMock()
        mock_user.id = uuid4()

        recall_mod = _get_recall_module()

        with (
            patch.object(
                _mod_search_methods,
                "authorized_search",
                AsyncMock(return_value=[mock_payload]),
            ),
        ):
            results = await recall_mod.recall(
                "test",
                query_type=SearchType.GRAPH_COMPLETION,
                session_id="s1",
                user=mock_user,
            )

        assert len(results) == 1
        assert results[0].source == "graph"
        assert results[0].text == "graph result"

    @pytest.mark.asyncio
    async def test_graph_recall_resolves_default_user(self):
        """Graph recall without an explicit user should use the default user."""
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_payload = SearchResultPayload(
            result_object="graph result", search_type=SearchType.GRAPH_COMPLETION
        )

        recall_mod = _get_recall_module()

        with (
            patch.object(recall_mod, "get_default_user", AsyncMock(return_value=mock_user)),
            patch.object(
                recall_mod,
                "set_session_user_context_variable",
                AsyncMock(),
            ) as set_user_context,
            patch.object(
                _mod_search_methods,
                "authorized_search",
                AsyncMock(return_value=[mock_payload]),
            ) as authorized_search,
        ):
            results = await recall_mod.recall(
                "test",
                query_type=SearchType.GRAPH_COMPLETION,
            )

        authorized_search.assert_awaited_once()
        assert authorized_search.await_args.kwargs["user"] is mock_user
        set_user_context.assert_awaited_once_with(mock_user)
        assert len(results) == 1
        assert results[0].source == "graph"
        assert results[0].text == "graph result"

    @pytest.mark.asyncio
    async def test_api_style_recall_kwargs_forward_search_options_once(self):
        """API recall kwargs should not duplicate or leak into authorized_search."""
        recall_mod = _get_recall_module()
        user = MagicMock()
        user.id = uuid4()
        dataset_id = uuid4()
        captured_kwargs = {}
        mock_payload = SearchResultPayload(
            result_object="graph result", search_type=SearchType.GRAPH_COMPLETION
        )

        async def dummy_authorized_search(**kwargs):
            captured_kwargs.update(kwargs)
            return [mock_payload]

        with patch.object(
            _mod_search_methods,
            "authorized_search",
            dummy_authorized_search,
        ):
            results = await recall_mod.recall(
                "test",
                query_type=SearchType.GRAPH_COMPLETION,
                user=user,
                datasets=["ignored-when-ids-exist"],
                dataset_ids=[dataset_id],
                system_prompt="Use concise answers.",
                node_name=["ImportantNode"],
                top_k=4,
                verbose=True,
                only_context=True,
                session_id="s1",
            )

        assert len(results) == 1
        assert captured_kwargs["user"] is user
        assert captured_kwargs["dataset_ids"] == [dataset_id]
        assert captured_kwargs["system_prompt"] == "Use concise answers."
        assert captured_kwargs["node_name"] == ["ImportantNode"]
        assert captured_kwargs["top_k"] == 4
        assert captured_kwargs["only_context"] is True
        assert captured_kwargs["session_id"] == "s1"
        assert "verbose" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_session_with_dataset_ids_also_runs_graph_search(self):
        """dataset_ids should count as graph scope, same as datasets by name."""
        recall_mod = _get_recall_module()
        user = MagicMock()
        user.id = uuid4()
        dataset_id = uuid4()

        session_entries = [
            ResponseQAEntry(
                time=datetime.utcnow().isoformat(),
                question="test",
                context="",
                answer="session result",
                source="session",
            )
        ]
        mock_payload = SearchResultPayload(
            result_object="graph result", search_type=SearchType.GRAPH_COMPLETION
        )

        with (
            patch.object(
                recall_mod,
                "_search_session",
                AsyncMock(return_value=session_entries),
            ) as search_session_mock,
            patch.object(
                recall_mod,
                "get_authorized_existing_datasets",
                AsyncMock(return_value=[SimpleNamespace(id=dataset_id)]),
            ),
            patch.object(
                _mod_search_methods,
                "authorized_search",
                AsyncMock(return_value=[mock_payload]),
            ) as authorized_search_mock,
            patch.object(
                _mod_query_router,
                "route_query",
                return_value=MagicMock(search_type=SearchType.GRAPH_COMPLETION),
            ),
        ):
            results = await recall_mod.recall(
                "test",
                session_id="s1",
                dataset_ids=[dataset_id],
                user=user,
            )

        assert [result.source for result in results] == ["session", "graph"]
        assert search_session_mock.await_args.kwargs["dataset_id"] == dataset_id
        assert authorized_search_mock.await_args.kwargs["dataset_ids"] == [dataset_id]
