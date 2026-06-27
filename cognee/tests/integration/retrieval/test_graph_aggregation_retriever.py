import pathlib
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

import cognee
from cognee.low_level import setup as setup_databases
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.modules.retrieval.graph_aggregation_retriever import (
    AggregationOperation,
    AggregationSpec,
    GraphAggregationRetriever,
)
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def _clear_engine_caches():
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()


@pytest_asyncio.fixture
async def setup_typed_graph():
    """Persist a tiny typed graph in the real graph DB: two EntityTypes (Issue, Pull
    Request) and three Entities, each linked to its type by an ``is_a`` edge. Nodes/edges
    are written straight to the graph engine (no cognify, no embeddings) so the test
    validates the aggregation against a real graph rather than mocks."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    cognee.config.system_root_directory(
        str(base_dir / ".cognee_system/test_graph_aggregation_retriever")
    )
    cognee.config.data_root_directory(
        str(base_dir / ".data_storage/test_graph_aggregation_retriever")
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    _clear_engine_caches()
    await setup_databases()

    issue_type = EntityType(name="Issue", description="A reported problem.")
    pr_type = EntityType(name="Pull Request", description="A proposed change.")
    login_bug = Entity(name="Login bug", description="an open login problem")
    crash = Entity(name="Crash", description="a closed crash report")
    fix_login = Entity(name="Fix login", description="an open pull request")

    graph_engine = await get_graph_engine()
    await graph_engine.add_nodes([issue_type, pr_type, login_bug, crash, fix_login])
    await graph_engine.add_edges(
        [
            (login_bug.id, issue_type.id, "is_a", {}),
            (crash.id, issue_type.id, "is_a", {}),
            (fix_login.id, pr_type.id, "is_a", {}),
            # Inter-entity relationships so "most connected" has real edges to rank.
            (login_bug.id, crash.id, "relates_to", {}),
            (login_bug.id, fix_login.id, "relates_to", {}),
        ]
    )

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        _clear_engine_caches()
    except Exception:
        pass


def _patch_spec(spec):
    return patch(
        "cognee.modules.retrieval.graph_aggregation_retriever.LLMGateway.acreate_structured_output",
        new=AsyncMock(return_value=spec),
    )


async def _run(spec):
    retriever = GraphAggregationRetriever()
    with _patch_spec(spec):
        retrieved = await retriever.get_retrieved_objects("question")
        context = await retriever.get_context_from_objects(retrieved_objects=retrieved)
        completion = await retriever.get_completion_from_context(
            retrieved_objects=retrieved, context=context
        )
    return retriever._compute_result(retrieved), context, completion


@pytest.mark.asyncio
async def test_count_excludes_pull_request_on_real_graph(setup_typed_graph):
    """The core of issue #3379: a Pull Request must not pollute an 'issues' count,
    verified end to end against the real graph database."""
    spec = AggregationSpec(operation=AggregationOperation.COUNT, target_type="issues")
    result, context, completion = await _run(spec)

    assert result["status"] == "ok"
    assert result["target_types"] == ["Issue"]
    # Two Issue entities, not three: the Pull Request is excluded structurally.
    assert result["count"] == 2
    assert isinstance(context, str) and "2" in context

    # The context/completion must satisfy SearchResultPayload's str|List[str] contract;
    # a dict would fail pydantic validation when the payload is constructed.
    payload = SearchResultPayload(
        result_object={"operation": result["operation"], "count": result["count"]},
        context=context,
        completion=completion,
        search_type=SearchType.GRAPH_AGGREGATION,
    )
    assert isinstance(payload.context, str)
    assert payload.completion == completion


@pytest.mark.asyncio
async def test_group_by_count_on_real_graph(setup_typed_graph):
    spec = AggregationSpec(operation=AggregationOperation.GROUP_BY_COUNT)
    result, _, _ = await _run(spec)

    assert result["status"] == "ok"
    assert result["groups"] == {"Issue": 2, "Pull Request": 1}
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_top_by_degree_on_real_graph(setup_typed_graph):
    spec = AggregationSpec(operation=AggregationOperation.TOP_BY_DEGREE)
    result, context, _ = await _run(spec)

    assert result["status"] == "ok"
    ranking = result["ranking"]
    ranked_names = [entry["name"] for entry in ranking]
    # EntityType nodes are never ranked as entities.
    assert "Issue" not in ranked_names and "Pull Request" not in ranked_names
    # "Login bug" links to its type plus two other entities -> most connected.
    assert ranking[0]["name"] == "Login bug"
    assert "Login bug" in context
