"""Full personalization loop on the REAL default stack (Ladybug + LanceDB + SQLite).

The unit suite fakes the graph engine everywhere, so nothing there proves the
end-to-end promise: remember content, rate a turn, run ``improve()`` and the
preference subgraph appears in the real graph — while never coming back as a
search result. This test drives the real remember -> rate -> improve -> recall
loop with real LLM and embedding calls, so it needs credentials: like the
neighbouring key-gated integration tests it is collected everywhere and skipped
when ``LLM_API_KEY`` is not set.
"""

from __future__ import annotations

import os
import pathlib
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.base_config import get_base_config
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.search.types import SearchType
from cognee.modules.user_preferences.constants import NEUTRAL_WEIGHT, PREFERS_RELATIONSHIP
from cognee.modules.user_preferences.store import preference_node_id
from cognee.modules.users.methods import get_default_user

try:
    import ladybug  # noqa: F401

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LADYBUG, reason="ladybug not installed"),
    pytest.mark.skipif(
        not os.getenv("LLM_API_KEY"),
        reason="LLM_API_KEY is not set; this test makes real LLM and embedding calls",
    ),
]


def _is_document_chunk(props: dict) -> bool:
    node_type = props.get("type")
    if isinstance(node_type, dict):
        return bool(node_type.get("DocumentChunk"))
    return node_type == "DocumentChunk"


@pytest_asyncio.fixture
async def default_stack_env(request, tmp_path, monkeypatch):
    """Clean default-stack environment with personalization switched on."""
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("PERSONALIZATION_ENABLED", "true")
    # Keep the loop to the calls the test needs: no per-turn feedback analysis.
    monkeypatch.setenv("AUTO_FEEDBACK", "false")
    get_base_config.cache_clear()

    root = pathlib.Path(tmp_path) / request.node.name

    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import create_cache_engine
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    def _clear_engine_caches():
        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
        get_cache_config.cache_clear()
        create_cache_engine.cache_clear()

    _clear_engine_caches()
    graph_db_config.set(None)
    vector_db_config.set(None)
    cognee.config.set_graph_db_config(
        {
            "graph_database_provider": "ladybug",
            "graph_dataset_database_handler": "ladybug",
        }
    )
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "lancedb",
            "vector_dataset_database_handler": "lancedb",
        }
    )
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.set_migration_db_config({"migration_db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass
    _clear_engine_caches()
    get_base_config.cache_clear()


async def test_preference_full_loop_isolated_from_search(default_stack_env):
    """remember -> rate a turn -> improve -> recall: the preference node exists
    in the real graph (with ``is_internal`` intact) and never surfaces in
    search results."""
    user = await get_default_user()
    dataset_name = f"pref_loop_{uuid4().hex[:8]}"

    await cognee.remember(
        (
            "The private codename for the first cognee personalization release is Velvet Otter. "
            "The launch principle is preferences before prompts."
        ),
        dataset_name=dataset_name,
        self_improvement=False,
    )

    dataset = (await get_datasets_by_name([dataset_name], user.id))[0]
    graph_engine = await get_graph_engine()

    nodes, _edges = await graph_engine.get_graph_data()
    chunk_ids = [str(node_id) for node_id, props in nodes if _is_document_chunk(props)]
    assert chunk_ids, "remember() left no DocumentChunk nodes to rate"
    rated_chunk_id = chunk_ids[0]

    # Rate a turn: a QA entry built from that chunk, rated 5/5 by the user.
    session_id = f"pref_loop_session_{uuid4().hex[:8]}"
    session_manager = get_session_manager()
    qa_id = await session_manager.add_qa(
        user_id=str(user.id),
        session_id=session_id,
        question="What is the private codename for the personalization release?",
        context="The private codename is Velvet Otter.",
        answer="The codename is Velvet Otter.",
        feedback_score=5,
        used_graph_element_ids={"node_ids": [rated_chunk_id], "edge_ids": []},
    )
    assert qa_id is not None, "session cache unavailable; cannot rate a turn"

    await cognee.improve(dataset=dataset_name, session_ids=[session_id], user=user)

    # The preference subgraph landed in the real graph, marked internal.
    pref_id = str(preference_node_id(str(user.id), str(dataset.id)))
    nodes, edges = await graph_engine.get_graph_data()
    pref_props = next((props for node_id, props in nodes if str(node_id) == pref_id), None)
    assert pref_props is not None, "improve() did not create the preference node"
    assert pref_props.get("is_internal") is True

    prefers_edges = {
        str(target_id): props
        for source_id, target_id, rel, props in edges
        if str(source_id) == pref_id and rel == PREFERS_RELATIONSHIP
    }
    assert rated_chunk_id in prefers_edges, "the rated chunk got no prefers edge"
    assert prefers_edges[rated_chunk_id].get("weight") > NEUTRAL_WEIGHT

    # The preference node never comes back as a search result — neither from
    # the vector lane (it is never embedded) nor from graph completion.
    chunk_results = await cognee.recall(
        "What is the launch principle?",
        query_type=SearchType.CHUNKS,
        datasets=[dataset_name],
        user=user,
    )
    assert chunk_results, "recall() over the rated content returned nothing"
    assert pref_id not in repr(chunk_results)

    completion_results = await cognee.recall(
        "What is the private codename for the personalization release?",
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[dataset_name],
        user=user,
    )
    assert completion_results
    assert pref_id not in repr(completion_results)
