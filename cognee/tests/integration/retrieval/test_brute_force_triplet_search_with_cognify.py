import pathlib

import pytest
import pytest_asyncio
import cognee

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points


@pytest_asyncio.fixture
async def clean_environment():
    """Configure isolated storage and ensure cleanup before/after."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_brute_force_triplet_search_e2e")
    data_directory_path = str(base_dir / ".data_storage/test_brute_force_triplet_search_e2e")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_brute_force_triplet_search_end_to_end(clean_environment):
    """Minimal end-to-end exercise of single and batch triplet search."""

    text = """
        Cognee is an open-source AI memory engine that structures data into searchable formats for use with AI agents.
        The company focuses on persistent memory systems using knowledge graphs and vector search.
        It is a Berlin-based startup building infrastructure for context-aware AI applications.
        NLP systems can use Cognee to store and retrieve structured information.
    """

    await cognee.add(text)
    await cognee.cognify()

    single_result = await brute_force_triplet_search(
        query="What can NLP systems use Cognee for?",
        top_k=1,
    )
    assert isinstance(single_result, list)
    assert single_result
    assert all(isinstance(edge, Edge) for edge in single_result)

    batch_queries = ["What is Cognee?", "What is the company's focus?"]
    batch_result = await brute_force_triplet_search(query_batch=batch_queries, top_k=1)

    assert isinstance(batch_result, list)
    assert len(batch_result) == len(batch_queries)
    assert all(isinstance(per_query, list) for per_query in batch_result)
    assert all(per_query for per_query in batch_result)
    assert all(isinstance(edge, Edge) for per_query in batch_result for edge in per_query)


@pytest.mark.asyncio
async def test_brute_force_triplet_search_feedback_does_not_override_missing_component_penalties(
    clean_environment,
):
    """With edge-only retrieval, feedback must not collapse node fallback penalties."""

    await setup()

    class Person(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    class Company(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    preferred_person = Person(name="Preferred Person", feedback_weight=0.95)
    preferred_company = Company(name="Preferred Company", feedback_weight=0.95)
    fallback_person = Person(name="Fallback Person", feedback_weight=0.05)
    fallback_company = Company(name="Fallback Company", feedback_weight=0.05)

    await add_data_points(
        [
            preferred_person,
            preferred_company,
            fallback_person,
            fallback_company,
        ],
        custom_edges=[
            (
                str(preferred_person.id),
                str(preferred_company.id),
                "works_for_different",
                {"relationship_name": "works_for", "feedback_weight": 0.95},
            ),
            (
                str(fallback_person.id),
                str(fallback_company.id),
                "works_for",
                {"relationship_name": "works_for", "feedback_weight": 0.05},
            ),
        ],
    )

    no_feedback_results = await brute_force_triplet_search(
        query="works_for",
        top_k=1,
        collections=["EdgeType_relationship_name"],
        feedback_influence=0.0,
    )

    full_feedback_results = await brute_force_triplet_search(
        query="works_for",
        top_k=1,
        collections=["EdgeType_relationship_name"],
        feedback_influence=1.0,
    )

    assert len(no_feedback_results) == 1
    assert len(full_feedback_results) == 1

    distance_only_edge = no_feedback_results[0]
    assert distance_only_edge.node1.attributes["name"] == "Fallback Person"
    assert distance_only_edge.node2.attributes["name"] == "Fallback Company"

    feedback_weighted_edge = full_feedback_results[0]
    assert feedback_weighted_edge.node1.attributes["name"] == "Fallback Person"
    assert feedback_weighted_edge.node2.attributes["name"] == "Fallback Company"
