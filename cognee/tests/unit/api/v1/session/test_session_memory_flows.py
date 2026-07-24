import asyncio
import importlib
from contextlib import contextmanager
from datetime import datetime
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


@contextmanager
def _patch_remember_startup():
    with (
        patch("cognee.modules.migrations.startup.run_migrations_and_block", new=AsyncMock()),
        patch("cognee.modules.engine.operations.setup.setup", new=AsyncMock()),
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
        """Session-stored result also tracks elapsed time."""
        mock_user = MagicMock()
        mock_user.id = "u1"

        mock_dataset = MagicMock()
        mock_dataset.id = uuid4()
        mock_dataset.name = "main_dataset"
        mock_dataset.owner_id = uuid4()

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.add_qa = AsyncMock()

        mock_db_context = MagicMock()
        mock_db_context.return_value.__aenter__ = AsyncMock(return_value=None)
        mock_db_context.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            _patch_remember_startup(),
            patch(
                "cognee.modules.users.methods.get_default_user",
                AsyncMock(return_value=mock_user),
            ),
            patch.object(
                _get_remember_module(),
                "resolve_authorized_user_datasets",
                AsyncMock(return_value=(mock_user, [mock_dataset])),
            ),
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                mock_db_context,
            ),
            patch.object(
                _mod_sm,
                "get_session_manager",
                return_value=mock_sm,
            ),
        ):
            from cognee.api.v1.remember.remember import RememberResult, remember

            result = await remember("test data", session_id="s1", self_improvement=False)

        assert isinstance(result, RememberResult)
        assert result.status == "session_stored"
        assert result.session_id == "s1"
        assert result.session_ids == ["s1"]
        assert result.dataset_id == str(mock_dataset.id)
        assert result.elapsed_seconds is not None


# ---------------------------------------------------------------------------
# _resolve_session_dataset: session writes target exactly one owned dataset
# ---------------------------------------------------------------------------


class TestResolveSessionDataset:
    @pytest.mark.asyncio
    async def test_returns_single_authorized_dataset(self):
        mock_user = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.id = uuid4()
        mock_dataset.owner_id = uuid4()

        with patch.object(
            _get_remember_module(),
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(mock_user, [mock_dataset])),
        ):
            from cognee.api.v1.remember.remember import _resolve_session_dataset

            user, dataset = await _resolve_session_dataset("main_dataset", mock_user)

        assert user is mock_user
        assert dataset is mock_dataset

    @pytest.mark.asyncio
    async def test_multiple_datasets_raises(self):
        mock_user = MagicMock()

        with patch.object(
            _get_remember_module(),
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(mock_user, [MagicMock(), MagicMock()])),
        ):
            from cognee.api.v1.remember.remember import _resolve_session_dataset

            with pytest.raises(ValueError, match="exactly one dataset"):
                await _resolve_session_dataset("main_dataset", mock_user)

    @pytest.mark.asyncio
    async def test_dataset_without_owner_raises(self):
        mock_user = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.owner_id = None

        with patch.object(
            _get_remember_module(),
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(mock_user, [mock_dataset])),
        ):
            from cognee.api.v1.remember.remember import _resolve_session_dataset

            with pytest.raises(ValueError, match="owner"):
                await _resolve_session_dataset("main_dataset", mock_user)


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
        assert authorized_search_mock.await_args.kwargs["dataset_ids"] == [dataset_id]


# ---------------------------------------------------------------------------
# _resolve_bound_session_dataset: session writes follow the session's binding
# ---------------------------------------------------------------------------


class TestResolveBoundSessionDataset:
    @staticmethod
    def _dataset(dataset_id=None):
        dataset = MagicMock()
        dataset.id = dataset_id or uuid4()
        dataset.owner_id = uuid4()
        return dataset

    @staticmethod
    @contextmanager
    def _patched(binding, resolved_dataset):
        """Patch the binding lookup and the dataset resolution around the helper."""
        mock_user = MagicMock()
        resolve = AsyncMock(return_value=(mock_user, [resolved_dataset]))
        with (
            patch(
                "cognee.modules.session_lifecycle.metrics.get_session_dataset",
                AsyncMock(return_value=binding),
            ),
            patch.object(_get_remember_module(), "resolve_authorized_user_datasets", resolve),
        ):
            yield mock_user, resolve

    @pytest.mark.asyncio
    async def test_unbound_session_resolves_caller_reference(self):
        from cognee.api.v1.remember.remember import _resolve_bound_session_dataset

        dataset = self._dataset()
        with self._patched(binding=None, resolved_dataset=dataset) as (mock_user, resolve):
            _, resolved = await _resolve_bound_session_dataset("s1", "main_dataset", mock_user)

        assert resolved is dataset
        resolve.assert_awaited_once_with("main_dataset", mock_user)

    @pytest.mark.asyncio
    async def test_bound_session_inherits_binding_for_default_reference(self):
        from cognee.api.v1.remember.remember import _resolve_bound_session_dataset

        bound_id = uuid4()
        dataset = self._dataset(bound_id)
        with self._patched(binding=(bound_id, uuid4()), resolved_dataset=dataset) as (
            mock_user,
            resolve,
        ):
            _, resolved = await _resolve_bound_session_dataset("s1", "main_dataset", mock_user)

        assert resolved is dataset
        resolve.assert_awaited_once_with(bound_id, mock_user)

    @pytest.mark.asyncio
    async def test_bound_session_rejects_other_dataset(self):
        from cognee.api.v1.remember.remember import _resolve_bound_session_dataset
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        dataset = self._dataset()
        with self._patched(binding=(uuid4(), uuid4()), resolved_dataset=dataset) as (
            mock_user,
            _,
        ):
            with pytest.raises(SessionDatasetMismatchError):
                await _resolve_bound_session_dataset("s1", "other_dataset", mock_user)

    @pytest.mark.asyncio
    async def test_bound_session_accepts_matching_dataset(self):
        from cognee.api.v1.remember.remember import _resolve_bound_session_dataset

        bound_id = uuid4()
        dataset = self._dataset(bound_id)
        with self._patched(binding=(bound_id, uuid4()), resolved_dataset=dataset) as (
            mock_user,
            _,
        ):
            _, resolved = await _resolve_bound_session_dataset("s1", "my_dataset", mock_user)

        assert resolved is dataset


# ---------------------------------------------------------------------------
# improve(session_ids=...): sessions cannot be bridged into a foreign dataset
# ---------------------------------------------------------------------------


class TestImproveSessionValidation:
    @staticmethod
    def _user():
        user = MagicMock()
        user.id = uuid4()
        return user

    @pytest.mark.asyncio
    async def test_nonexistent_dataset_rejects_bound_sessions(self):
        """Downstream stages create missing datasets — a bound session must not
        slip through the not-found branch into a brand-new dataset."""
        from cognee.api.v1.improve.improve import _check_sessions_belong_to_dataset
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        with (
            patch(
                "cognee.modules.data.methods.get_authorized_existing_datasets",
                AsyncMock(return_value=[]),
            ),
            patch(
                "cognee.modules.session_lifecycle.metrics.get_session_dataset",
                AsyncMock(return_value=(uuid4(), uuid4())),
            ),
        ):
            with pytest.raises(SessionDatasetMismatchError):
                await _check_sessions_belong_to_dataset("new_dataset", ["s1"], self._user())

    @pytest.mark.asyncio
    async def test_nonexistent_dataset_allows_unbound_sessions(self):
        from cognee.api.v1.improve.improve import _check_sessions_belong_to_dataset

        with (
            patch(
                "cognee.modules.data.methods.get_authorized_existing_datasets",
                AsyncMock(return_value=[]),
            ),
            patch(
                "cognee.modules.session_lifecycle.metrics.get_session_dataset",
                AsyncMock(return_value=None),
            ),
        ):
            await _check_sessions_belong_to_dataset("new_dataset", ["s1"], self._user())

    @pytest.mark.asyncio
    async def test_existing_dataset_delegates_to_binding_check(self):
        from cognee.api.v1.improve.improve import _check_sessions_belong_to_dataset

        dataset = MagicMock()
        dataset.id = uuid4()
        user = self._user()
        check = AsyncMock()
        with (
            patch(
                "cognee.modules.data.methods.get_authorized_existing_datasets",
                AsyncMock(return_value=[dataset]),
            ),
            patch(
                "cognee.modules.session_lifecycle.metrics.check_session_dataset_binding",
                check,
            ),
        ):
            await _check_sessions_belong_to_dataset("docs", ["s1", "s2"], user)

        assert check.await_count == 2
        check.assert_awaited_with(session_id="s2", user_id=user.id, dataset_id=dataset.id)


# ---------------------------------------------------------------------------
# Agentic retriever: a session-dataset mismatch surfaces instead of being
# swallowed by the generic storage-failure handler
# ---------------------------------------------------------------------------


class TestAgenticSessionMismatchSurfaces:
    @staticmethod
    def _retriever():
        from cognee.modules.retrieval.agentic_retriever import AgenticRetriever

        user = MagicMock()
        user.id = uuid4()
        retriever = AgenticRetriever(user=user, dataset_id=uuid4())
        retriever.session_id = "s1"
        retriever._use_session_cache = lambda: True
        return retriever

    @pytest.mark.asyncio
    async def test_mismatch_reraises(self):
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        retriever = self._retriever()
        manager = MagicMock()
        manager.is_available = True
        manager.add_qa = AsyncMock(side_effect=SessionDatasetMismatchError("s1", uuid4(), uuid4()))

        with patch(
            "cognee.infrastructure.session.get_session_manager.get_session_manager",
            return_value=manager,
        ):
            with pytest.raises(SessionDatasetMismatchError):
                await retriever._store_session_qa("q", "ctx", "a", triplets=[])

    @pytest.mark.asyncio
    async def test_other_storage_errors_stay_swallowed(self):
        retriever = self._retriever()
        manager = MagicMock()
        manager.is_available = True
        manager.add_qa = AsyncMock(side_effect=RuntimeError("cache down"))

        with patch(
            "cognee.infrastructure.session.get_session_manager.get_session_manager",
            return_value=manager,
        ):
            await retriever._store_session_qa("q", "ctx", "a", triplets=[])
