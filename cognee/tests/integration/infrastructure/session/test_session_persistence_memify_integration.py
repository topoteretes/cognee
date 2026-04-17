import asyncio
import os
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.low_level import setup as cognee_setup
from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
    persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
)
from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
    persist_sessions_in_knowledge_graph_pipeline,
)
from cognee.modules.engine.models import NodeSet
from cognee.modules.users.methods import get_default_user


def _reset_cache_backend_caches():
    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine

    get_cache_config.cache_clear()
    create_cache_engine.cache_clear()


async def _reset_engines_and_prune() -> None:
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so all tests and the env fixture share one loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class _InMemoryRedisList:
    """Minimal in-memory Redis list emulation for Redis-backed session tests."""

    def __init__(self):
        self.data: dict[str, list[str]] = {}

    async def rpush(self, key: str, *vals: str):
        self.data.setdefault(key, []).extend(vals)

    async def lrange(self, key: str, start: int, end: int):
        lst = self.data.get(key, [])
        s = start if start >= 0 else len(lst) + start
        e = (end + 1) if end >= 0 else len(lst) + end + 1
        return lst[s:e]

    async def lindex(self, key: str, idx: int):
        lst = self.data.get(key, [])
        return lst[idx] if -len(lst) <= idx < len(lst) else None

    async def lset(self, key: str, idx: int, val: str):
        self.data[key][idx] = val

    async def delete(self, key: str):
        return 1 if self.data.pop(key, None) is not None else 0

    async def expire(self, key: str, ttl: int):
        pass

    async def flushdb(self):
        self.data.clear()


def _count_document_chunks(nodes) -> int:
    document_chunk_count = 0
    for _node_id, props in nodes:
        node_type = props.get("type")
        if isinstance(node_type, dict) and node_type.get("DocumentChunk"):
            document_chunk_count += 1
        elif node_type == "DocumentChunk":
            document_chunk_count += 1
    return document_chunk_count


@pytest_asyncio.fixture(scope="module")
async def session_persistence_env(event_loop):
    """Clean cognee env with one dataset (add + cognify); shared by all tests in this module."""
    with (
        patch.dict(
            os.environ,
            {"CACHE_BACKEND": "fs", "COGNEE_SKIP_CONNECTION_TEST": "true"},
            clear=False,
        ),
        tempfile.TemporaryDirectory(prefix="cognee_session_persistence_system_") as system_path,
        tempfile.TemporaryDirectory(prefix="cognee_session_persistence_data_") as data_path,
    ):
        pytest.importorskip("kuzu")
        _reset_cache_backend_caches()

        vector_db_config.set(None)
        graph_db_config.set(None)
        cognee.config.set_graph_database_provider("kuzu")
        cognee.config.set_vector_db_config(
            {
                "vector_db_provider": "lancedb",
                "vector_dataset_database_handler": "lancedb",
            }
        )
        cognee.config.set_relational_db_config({"db_provider": "sqlite"})
        cognee.config.set_migration_db_config({"migration_db_provider": "sqlite"})
        cognee.config.system_root_directory(system_path)
        cognee.config.data_root_directory(data_path)
        cognee.config.set_vector_db_url(
            str(pathlib.Path(system_path) / "databases" / "cognee.lancedb")
        )

        await _reset_engines_and_prune()
        await cognee_setup()

        dataset_name = "session_persist_integration"
        await cognee.add("Cognee builds knowledge graphs from text.", dataset_name=dataset_name)
        await cognee.cognify(datasets=[dataset_name])

        yield dataset_name

        try:
            await _reset_engines_and_prune()
        except Exception:
            pass
        finally:
            _reset_cache_backend_caches()


@pytest.fixture(params=["fs", "redis"])
def session_manager_with_qa(request):
    """
    SessionManager backed by either FsCache or in-memory Redis.
    Tests run twice (once per backend).
    """
    backend = request.param
    if backend == "fs":
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
                return_value={"data_root_directory": tmpdir},
            ):
                from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                    FSCacheAdapter,
                )

                adapter = FSCacheAdapter()
                sm = SessionManager(cache_engine=adapter)
                yield sm, adapter
                adapter.cache.close()
    else:
        store = _InMemoryRedisList()
        patch_mod = "cognee.infrastructure.databases.cache.redis.RedisAdapter"
        with (
            patch(f"{patch_mod}.redis.Redis", return_value=MagicMock(ping=MagicMock())),
            patch(f"{patch_mod}.aioredis.Redis", return_value=store),
        ):
            from cognee.infrastructure.databases.cache.redis.RedisAdapter import (
                RedisAdapter,
            )

            adapter = RedisAdapter(host="localhost", port=6379)
            sm = SessionManager(cache_engine=adapter)
            yield sm, adapter


@pytest.mark.asyncio
async def test_persist_sessions_in_knowledge_graph_uses_session_manager(
    session_persistence_env,
    session_manager_with_qa,
):
    """
    Session persistence memify pipeline reads from SessionManager and writes to the graph.

    - Populate SessionManager with Q&A for a session.
    - Patch extract_user_sessions to use this SessionManager.
    - Run persist_sessions_in_knowledge_graph_pipeline.
    - Assert session content appears in the graph (DocumentChunk count increases).
    """
    dataset_name = session_persistence_env
    session_manager, _adapter = session_manager_with_qa

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_qa(
        user_id=user_id,
        question="What does Cognee do?",
        context="Intro to Cognee",
        answer="Cognee builds knowledge graphs from text.",
        session_id="integration_test_session",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=["integration_test_session"],
            dataset=dataset_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()

    document_chunk_count = _count_document_chunks(nodes)

    assert document_chunk_count >= 2, (
        "Expected at least 2 DocumentChunks (1 from initial add+cognify, 1 from session). "
        f"Got {document_chunk_count} DocumentChunk nodes, total nodes: {len(nodes)}"
    )


@pytest.mark.asyncio
async def test_persist_sessions_multiple_sessions_via_session_manager(
    session_persistence_env,
    session_manager_with_qa,
):
    """Persist two sessions from SessionManager; both are cognified into the graph."""
    dataset_name = session_persistence_env
    session_manager, _adapter = session_manager_with_qa

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_qa(
        user_id=user_id,
        question="First question?",
        context="C1",
        answer="First answer.",
        session_id="session_a",
    )
    await session_manager.add_qa(
        user_id=user_id,
        question="Second question?",
        context="C2",
        answer="Second answer.",
        session_id="session_b",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=["session_a", "session_b"],
            dataset=dataset_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_graph_data()

    assert len(nodes) >= 3, (
        f"Expected at least 3 nodes (1 from initial doc + 2 from sessions). Got {len(nodes)} nodes."
    )


@pytest.mark.asyncio
async def test_persist_agent_trace_feedbacks_in_knowledge_graph_uses_session_manager(
    session_persistence_env,
    session_manager_with_qa,
):
    """Persist session-backed agent trace feedback into the knowledge graph."""
    dataset_name = session_persistence_env
    session_manager, adapter = session_manager_with_qa
    backend_label = type(adapter).__name__.lower()
    session_id = f"integration_trace_session_{backend_label}"
    node_set_name = f"agent_trace_feedbacks_{backend_label}"

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"draft_plan_{backend_label}",
        status="success",
        generate_feedback_with_llm=False,
        session_id=session_id,
        method_return_value="draft ready",
    )
    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"write_summary_{backend_label}",
        status="error",
        generate_feedback_with_llm=False,
        session_id=session_id,
        error_message=f"missing data {backend_label}",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[session_id],
            dataset=dataset_name,
            node_set_name=node_set_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_nodeset_subgraph(NodeSet, [node_set_name])

    assert any(node[1].get("name") == node_set_name for node in nodes)
    assert nodes or edges


@pytest.mark.asyncio
async def test_persist_agent_trace_return_values_in_knowledge_graph_uses_session_manager(
    session_persistence_env,
    session_manager_with_qa,
):
    """Persist raw agent trace return values into the knowledge graph."""
    dataset_name = session_persistence_env
    session_manager, adapter = session_manager_with_qa
    backend_label = type(adapter).__name__.lower()
    session_id = f"integration_trace_returns_{backend_label}"
    node_set_name = f"agent_trace_returns_{backend_label}"

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"draft_plan_returns_{backend_label}",
        status="success",
        generate_feedback_with_llm=False,
        session_id=session_id,
        method_return_value={"backend": backend_label, "result": "draft ready"},
    )
    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"write_summary_returns_{backend_label}",
        status="error",
        generate_feedback_with_llm=False,
        session_id=session_id,
        error_message=f"missing data {backend_label}",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[session_id],
            dataset=dataset_name,
            node_set_name=node_set_name,
            raw_trace_content=True,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_nodeset_subgraph(NodeSet, [node_set_name])

    assert any(node[1].get("name") == node_set_name for node in nodes)
    assert nodes or edges


@pytest.mark.asyncio
async def test_persist_agent_trace_return_values_skips_empty_return_values(
    session_persistence_env,
    session_manager_with_qa,
):
    """Sessions with only empty raw return values should not add new graph content."""
    dataset_name = session_persistence_env
    session_manager, adapter = session_manager_with_qa
    backend_label = type(adapter).__name__.lower()
    session_id = f"integration_empty_trace_returns_{backend_label}"
    node_set_name = f"empty_trace_returns_{backend_label}"

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"draft_plan_empty_returns_{backend_label}",
        status="success",
        generate_feedback_with_llm=False,
        session_id=session_id,
        method_return_value="   ",
    )
    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"write_summary_empty_returns_{backend_label}",
        status="error",
        generate_feedback_with_llm=False,
        session_id=session_id,
        error_message=f"missing data {backend_label}",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[session_id],
            dataset=dataset_name,
            node_set_name=node_set_name,
            raw_trace_content=True,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_nodeset_subgraph(NodeSet, [node_set_name])

    assert nodes == []
    assert edges == []


@pytest.mark.asyncio
async def test_persist_agent_trace_feedbacks_attaches_content_to_requested_node_set(
    session_persistence_env,
    session_manager_with_qa,
):
    """Persisted trace content should be reachable through the configured node set."""
    dataset_name = session_persistence_env
    session_manager, adapter = session_manager_with_qa
    backend_label = type(adapter).__name__.lower()
    session_id = f"integration_trace_nodeset_{backend_label}"
    node_set_name = f"custom_agent_trace_feedbacks_{backend_label}"

    user = await get_default_user()
    user_id = str(user.id)

    await session_manager.add_agent_trace_step(
        user_id=user_id,
        origin_function=f"draft_plan_nodeset_{backend_label}",
        status="success",
        generate_feedback_with_llm=False,
        session_id=session_id,
        method_return_value=f"draft ready for {backend_label}",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[session_id],
            dataset=dataset_name,
            node_set_name=node_set_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, _edges = await graph_engine.get_nodeset_subgraph(NodeSet, [node_set_name])

    assert any(node[1].get("name") == node_set_name for node in nodes)
    assert _count_document_chunks(nodes) >= 1


@pytest.mark.asyncio
async def test_persist_agent_trace_feedbacks_skips_empty_feedbacks(
    session_persistence_env,
    session_manager_with_qa,
):
    """Sessions with empty feedback summaries should not add new graph content."""
    dataset_name = session_persistence_env
    session_manager, adapter = session_manager_with_qa
    backend_label = type(adapter).__name__.lower()
    session_id = f"integration_empty_trace_feedbacks_{backend_label}"
    node_set_name = f"empty_trace_feedbacks_{backend_label}"

    user = await get_default_user()
    user_id = str(user.id)

    await adapter.append_agent_trace_step(
        user_id=user_id,
        session_id=session_id,
        trace_id=f"{session_id}_step_1",
        origin_function=f"draft_plan_empty_feedbacks_{backend_label}",
        status="success",
        method_return_value="draft ready",
        session_feedback="   ",
    )
    await adapter.append_agent_trace_step(
        user_id=user_id,
        session_id=session_id,
        trace_id=f"{session_id}_step_2",
        origin_function=f"write_summary_empty_feedbacks_{backend_label}",
        status="error",
        error_message=f"missing data {backend_label}",
        session_feedback="",
    )

    extract_module = sys.modules["cognee.tasks.memify.extract_agent_trace_feedbacks"]
    with patch.object(extract_module, "get_session_manager", return_value=session_manager):
        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=[session_id],
            dataset=dataset_name,
            node_set_name=node_set_name,
            run_in_background=False,
        )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_nodeset_subgraph(NodeSet, [node_set_name])

    assert nodes == []
    assert edges == []
