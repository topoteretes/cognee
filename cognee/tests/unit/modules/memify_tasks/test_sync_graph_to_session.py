import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from cognee.tasks.memify.sync_graph_to_session import (
    sync_graph_to_session,
    _edge_to_text,
    _checkpoint_key,
)

sync_module = sys.modules["cognee.tasks.memify.sync_graph_to_session"]

# Patch paths for lazy imports inside sync_graph_to_session()
_PATCH_GET_CACHE = "cognee.infrastructure.databases.cache.get_cache_engine.get_cache_engine"
_PATCH_GET_SM = "cognee.infrastructure.session.get_session_manager.get_session_manager"
_PATCH_GET_REL = "cognee.tasks.memify.sync_graph_to_session.get_relational_engine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(id_, label=None, type_="Entity"):
    node = MagicMock()
    node.id = id_
    node.label = label
    node.type = type_
    return node


def _make_edge(src_id, dst_id, rel_name="related_to", created_at=None):
    edge = MagicMock()
    edge.source_node_id = src_id
    edge.destination_node_id = dst_id
    edge.relationship_name = rel_name
    edge.created_at = created_at or datetime.now(timezone.utc)
    edge.dataset_id = uuid4()
    return edge


def _make_session_manager(graph_context=""):
    sm = MagicMock()
    sm.is_available = True
    sm.get_graph_context = AsyncMock(return_value=graph_context)
    sm.set_graph_context = AsyncMock()
    return sm


def _make_cache_engine(checkpoint_value=None):
    engine = MagicMock()
    engine.async_redis = MagicMock()
    engine.async_redis.get = AsyncMock(return_value=checkpoint_value)
    engine.async_redis.set = AsyncMock()
    return engine


def _mock_db_engine_returning(edges, nodes):
    """Build a mock relational engine that returns edges then nodes, then empty on next batch."""
    mock_db_engine = MagicMock()
    mock_session = MagicMock()

    edge_scalars = MagicMock()
    edge_scalars.all.return_value = edges
    node_scalars = MagicMock()
    node_scalars.all.return_value = nodes
    empty_scalars = MagicMock()
    empty_scalars.all.return_value = []

    call_count = 0

    async def mock_scalars_fn(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return edge_scalars
        elif call_count == 2:
            return node_scalars
        return empty_scalars

    mock_session.scalars = mock_scalars_fn
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_db_engine.get_async_session.return_value = mock_session
    return mock_db_engine


def _mock_db_engine_empty():
    return _mock_db_engine_returning([], [])


# ---------------------------------------------------------------------------
# _edge_to_text
# ---------------------------------------------------------------------------


class TestEdgeToText:
    def test_renders_labeled_nodes(self):
        n1 = _make_node("a", label="Alice")
        n2 = _make_node("b", label="Bob")
        edge = _make_edge("a", "b", "knows")
        result = _edge_to_text(edge, {"a": n1, "b": n2})
        assert result == "Alice —[knows]→ Bob"

    def test_falls_back_to_type_when_no_label(self):
        n1 = _make_node("a", label=None, type_="Person")
        n2 = _make_node("b", label=None, type_="City")
        edge = _make_edge("a", "b", "lives_in")
        result = _edge_to_text(edge, {"a": n1, "b": n2})
        assert result == "Person —[lives_in]→ City"

    def test_returns_none_when_node_missing(self):
        n1 = _make_node("a", label="Alice")
        edge = _make_edge("a", "b", "knows")
        assert _edge_to_text(edge, {"a": n1}) is None

    def test_default_relationship_name(self):
        n1 = _make_node("a", label="X")
        n2 = _make_node("b", label="Y")
        edge = _make_edge("a", "b")
        edge.relationship_name = None
        result = _edge_to_text(edge, {"a": n1, "b": n2})
        assert "related_to" in result


# ---------------------------------------------------------------------------
# sync_graph_to_session — no new edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_no_new_edges():
    dataset_id = uuid4()
    sm = _make_session_manager()
    cache_engine = _make_cache_engine()
    db_engine = _mock_db_engine_empty()

    with (
        patch(_PATCH_GET_SM, return_value=sm),
        patch(_PATCH_GET_CACHE, return_value=cache_engine),
        patch(_PATCH_GET_REL, return_value=db_engine),
    ):
        result = await sync_graph_to_session(
            user_id="u1", session_id="s1", dataset_id=dataset_id,
        )

    assert result["synced"] == 0
    sm.set_graph_context.assert_not_called()


# ---------------------------------------------------------------------------
# sync_graph_to_session — cache unavailable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_cache_unavailable():
    sm = MagicMock()
    sm.is_available = False

    with (
        patch(_PATCH_GET_SM, return_value=sm),
        patch(_PATCH_GET_CACHE, return_value=None),
    ):
        result = await sync_graph_to_session(
            user_id="u1", session_id="s1", dataset_id=uuid4(),
        )

    assert result["synced"] == 0


# ---------------------------------------------------------------------------
# sync_graph_to_session — merges with existing context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_merges_with_existing():
    dataset_id = uuid4()
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    n1 = _make_node("a", label="Alice")
    n2 = _make_node("b", label="Bob")
    edge = _make_edge("a", "b", "knows", created_at=ts)

    sm = _make_session_manager(graph_context="Carol —[likes]→ Dave")
    cache_engine = _make_cache_engine()
    db_engine = _mock_db_engine_returning([edge], [n1, n2])

    with (
        patch(_PATCH_GET_SM, return_value=sm),
        patch(_PATCH_GET_CACHE, return_value=cache_engine),
        patch(_PATCH_GET_REL, return_value=db_engine),
    ):
        result = await sync_graph_to_session(
            user_id="u1", session_id="s1", dataset_id=dataset_id,
        )

    assert result["synced"] == 1
    assert result["total"] == 2

    context = sm.set_graph_context.call_args.kwargs["context"]
    assert "Carol —[likes]→ Dave" in context
    assert "Alice —[knows]→ Bob" in context


# ---------------------------------------------------------------------------
# sync_graph_to_session — max_lines cap drops oldest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_caps_at_max_lines():
    dataset_id = uuid4()
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    nodes = {f"n{i}": _make_node(f"n{i}", label=f"Node{i}") for i in range(4)}
    edges = [
        _make_edge("n0", "n1", "r0", created_at=ts),
        _make_edge("n1", "n2", "r1", created_at=ts),
        _make_edge("n2", "n3", "r2", created_at=ts),
    ]

    # Existing has 2 lines, new adds 3, cap at 3 → oldest 2 dropped
    existing = "OldA —[x]→ OldB\nOldC —[y]→ OldD"
    sm = _make_session_manager(graph_context=existing)
    cache_engine = _make_cache_engine()
    db_engine = _mock_db_engine_returning(edges, list(nodes.values()))

    with (
        patch(_PATCH_GET_SM, return_value=sm),
        patch(_PATCH_GET_CACHE, return_value=cache_engine),
        patch(_PATCH_GET_REL, return_value=db_engine),
    ):
        result = await sync_graph_to_session(
            user_id="u1", session_id="s1", dataset_id=dataset_id, max_lines=3,
        )

    assert result["total"] == 3

    context = sm.set_graph_context.call_args.kwargs["context"]
    lines = context.split("\n")
    assert len(lines) == 3
    # Old lines should be dropped (they were at the front)
    assert "OldA —[x]→ OldB" not in lines
    assert "OldC —[y]→ OldD" not in lines


# ---------------------------------------------------------------------------
# _checkpoint_key format
# ---------------------------------------------------------------------------


def test_checkpoint_key_format():
    key = _checkpoint_key("u1", "d1", "s1")
    assert key == "graph_sync_checkpoint:u1:d1:s1"


# ---------------------------------------------------------------------------
# SessionManager.get/set_graph_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_manager_graph_context_roundtrip():
    """Test get/set graph context via SessionManager with a mock Redis cache."""
    from cognee.infrastructure.session.session_manager import SessionManager

    mock_cache = MagicMock()
    mock_cache.async_redis = MagicMock()

    stored = {}

    async def mock_set(key, value):
        stored[key] = value

    async def mock_get(key):
        return stored.get(key)

    async def mock_expire(key, ttl):
        pass

    mock_cache.async_redis.set = mock_set
    mock_cache.async_redis.get = mock_get
    mock_cache.async_redis.expire = mock_expire
    mock_cache.session_ttl_seconds = 3600

    sm = SessionManager(cache_engine=mock_cache)

    ctx = await sm.get_graph_context(user_id="u1", session_id="s1")
    assert ctx == ""

    await sm.set_graph_context(
        user_id="u1", session_id="s1", context="Alice —[knows]→ Bob"
    )

    ctx = await sm.get_graph_context(user_id="u1", session_id="s1")
    assert ctx == "Alice —[knows]→ Bob"

    await sm.set_graph_context(
        user_id="u1",
        session_id="s1",
        context="Alice —[knows]→ Bob\nBob —[likes]→ Carol",
    )
    ctx = await sm.get_graph_context(user_id="u1", session_id="s1")
    assert "Bob —[likes]→ Carol" in ctx


@pytest.mark.asyncio
async def test_session_manager_graph_context_unavailable():
    """When cache is None, get returns empty, set is a no-op."""
    from cognee.infrastructure.session.session_manager import SessionManager

    sm = SessionManager(cache_engine=None)
    assert await sm.get_graph_context(user_id="u1", session_id="s1") == ""
    await sm.set_graph_context(user_id="u1", session_id="s1", context="test")


# ---------------------------------------------------------------------------
# Graph context injection into completion prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_context_prepended_to_completion():
    """Verify generate_completion_with_session includes graph context in the prompt."""
    from cognee.infrastructure.session.session_manager import SessionManager

    mock_cache = MagicMock()
    mock_cache.async_redis = MagicMock()
    mock_cache.get_latest_qa_entries = AsyncMock(return_value=[])
    mock_cache.get_all_qa_entries = AsyncMock(return_value=[])
    mock_cache.create_qa_entry = AsyncMock()

    graph_ctx = "Alice —[knows]→ Bob"

    async def mock_get(key):
        if "graph_knowledge:" in key:
            return graph_ctx.encode()
        return None

    mock_cache.async_redis.get = mock_get
    mock_cache.session_ttl_seconds = 3600

    sm = SessionManager(cache_engine=mock_cache)

    mock_user = MagicMock()
    mock_user.id = "u1"

    captured_history = {}

    async def mock_generate(**kwargs):
        captured_history["value"] = kwargs.get("conversation_history", "")
        return ("answer", "", None)

    with (
        patch(
            "cognee.infrastructure.session.session_manager.session_user"
        ) as mock_session_user,
        patch(
            "cognee.infrastructure.session.session_manager.CacheConfig"
        ) as MockCacheConfig,
        patch(
            "cognee.infrastructure.session.session_manager."
            "generate_session_completion_with_optional_summary",
            side_effect=mock_generate,
        ),
    ):
        mock_session_user.get.return_value = mock_user
        MockCacheConfig.return_value.caching = True
        MockCacheConfig.return_value.auto_feedback = False

        await sm.generate_completion_with_session(
            session_id="s1",
            query="test question",
            context="some context",
            user_prompt_path="test.txt",
            system_prompt_path="test.txt",
        )

    assert "Background knowledge from the knowledge graph:" in captured_history["value"]
    assert "Alice —[knows]→ Bob" in captured_history["value"]


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
        patch("cognee.api.v1.add.add", AsyncMock()),
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={"status": "ok"})),
        patch("cognee.api.v2.improve.improve", mock_improve),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(return_value=mock_user),
        ),
    ):
        from cognee.api.v2.remember.remember import remember

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
        patch("cognee.api.v1.add.add", AsyncMock()),
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={"status": "ok"})),
        patch("cognee.api.v2.improve.improve", mock_improve),
        patch(
            "cognee.modules.users.methods.get_default_user",
            AsyncMock(return_value=mock_user),
        ),
    ):
        from cognee.api.v2.remember.remember import remember

        await remember("test data", self_improvement=True)

    assert len(improve_calls) == 1
    assert "session_ids" not in improve_calls[0]


# ---------------------------------------------------------------------------
# RememberResult
# ---------------------------------------------------------------------------


class TestRememberResult:
    def test_repr_basic(self):
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="test")
        assert "completed" in repr(r)
        assert "test" in repr(r)

    def test_repr_includes_elapsed(self):
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="test")
        r.elapsed_seconds = 4.2
        assert "elapsed=4.2s" in repr(r)

    def test_repr_includes_error(self):
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="errored", dataset_name="test")
        r.error = "something broke"
        assert "something broke" in repr(r)

    def test_bool_completed_is_true(self):
        from cognee.api.v2.remember.remember import RememberResult

        assert bool(RememberResult(status="completed", dataset_name="x"))
        assert bool(RememberResult(status="session_stored", dataset_name="x"))

    def test_bool_running_is_false(self):
        from cognee.api.v2.remember.remember import RememberResult

        assert not bool(RememberResult(status="running", dataset_name="x"))
        assert not bool(RememberResult(status="errored", dataset_name="x"))

    def test_done_without_task(self):
        from cognee.api.v2.remember.remember import RememberResult

        assert RememberResult(status="completed", dataset_name="x").done is True
        assert RememberResult(status="errored", dataset_name="x").done is True
        assert RememberResult(status="running", dataset_name="x").done is False

    def test_done_with_task(self):
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="x")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        r._task = mock_task
        assert r.done is False

        mock_task.done.return_value = True
        assert r.done is True

    @pytest.mark.asyncio
    async def test_await_completed_returns_self(self):
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="completed", dataset_name="x")
        result = await r
        assert result is r

    @pytest.mark.asyncio
    async def test_await_background_task(self):
        """Awaiting a result with a background task waits for the task."""
        from cognee.api.v2.remember.remember import RememberResult

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
        from cognee.api.v2.remember.remember import RememberResult

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
        from cognee.api.v2.remember.remember import RememberResult

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
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="test")

        mock_run_info = MagicMock()
        mock_run_info.status = "PipelineRunErrored"

        r._resolve({UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"): mock_run_info})
        assert r.status == "errored"

    def test_fail_sets_error_and_elapsed(self):
        from cognee.api.v2.remember.remember import RememberResult

        r = RememberResult(status="running", dataset_name="test")
        r._fail(ValueError("test error"))
        assert r.status == "errored"
        assert r.error == "test error"
        assert r.elapsed_seconds is not None

    @pytest.mark.asyncio
    async def test_remember_returns_remember_result(self):
        """Blocking remember() returns a properly resolved RememberResult."""
        from cognee.api.v2.remember.remember import RememberResult

        mock_user = MagicMock()
        mock_user.id = "u1"

        with (
            patch("cognee.api.v1.add.add", AsyncMock()),
            patch(
                "cognee.api.v1.cognify.cognify",
                AsyncMock(return_value={}),
            ),
            patch("cognee.api.v2.improve.improve", AsyncMock()),
            patch(
                "cognee.modules.users.methods.get_default_user",
                AsyncMock(return_value=mock_user),
            ),
        ):
            from cognee.api.v2.remember.remember import remember

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

        mock_sm = MagicMock()
        mock_sm.is_available = True
        mock_sm.add_qa = AsyncMock()

        with (
            patch(
                "cognee.modules.users.methods.get_default_user",
                AsyncMock(return_value=mock_user),
            ),
            patch(
                "cognee.infrastructure.session.get_session_manager.get_session_manager",
                return_value=mock_sm,
            ),
        ):
            from cognee.api.v2.remember.remember import remember, RememberResult

            result = await remember("test data", session_id="s1")

        assert isinstance(result, RememberResult)
        assert result.status == "session_stored"
        assert result.session_id == "s1"
        assert result.elapsed_seconds is not None
