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
async def test_node_feedback_does_not_modify_penalty_placeholder_in_edge_only_retrieval(
    clean_environment,
):
    """The penalty placeholder assigned to a node with no vector match in the
    searched collection is not modified by the node's feedback_weight.
    Feedback only blends real cosine distances; placeholder values bypass
    that blend.

    Setup: two candidate edges, identical in every factor that the scorer
    weighs except the node-level feedback_weight. Both edges share the
    same explicit edge_text (so the EdgeType vector is shared and edge
    distance is tied by construction) and the same edge feedback_weight.
    Limiting retrieval to the EdgeType collection forces both candidates'
    nodes to receive the penalty placeholder.

    If the placeholder were affected by feedback_weight, cranking
    feedback_influence from 0 to 1 would let the high-node-feedback pair
    overtake the low-node-feedback pair. The penalty floor must prevent
    that, so the same edge must win in both runs.
    """

    await setup()

    class Person(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    class Company(DataPoint):
        name: str
        metadata: dict = {"index_fields": ["name"]}

    person_a = Person(name="Person A", feedback_weight=0.95)
    company_a = Company(name="Company A", feedback_weight=0.95)
    person_b = Person(name="Person B", feedback_weight=0.05)
    company_b = Company(name="Company B", feedback_weight=0.05)

    shared_edge_text = "Some person works for some company."
    shared_edge_feedback = 0.5

    await add_data_points(
        [person_a, company_a, person_b, company_b],
        custom_edges=[
            (
                str(person_a.id),
                str(company_a.id),
                "works_for",
                {
                    "edge_text": shared_edge_text,
                    "feedback_weight": shared_edge_feedback,
                },
            ),
            (
                str(person_b.id),
                str(company_b.id),
                "works_for",
                {
                    "edge_text": shared_edge_text,
                    "feedback_weight": shared_edge_feedback,
                },
            ),
        ],
    )

    search_kwargs = dict(
        query=shared_edge_text,
        top_k=1,
        collections=["EdgeType_relationship_name"],
    )

    top_with_no_feedback = await brute_force_triplet_search(**search_kwargs, feedback_influence=0.0)
    top_with_full_feedback = await brute_force_triplet_search(
        **search_kwargs, feedback_influence=1.0
    )

    assert len(top_with_no_feedback) == 1
    assert len(top_with_full_feedback) == 1

    winner_without_feedback = top_with_no_feedback[0].node1.attributes["name"]
    winner_with_full_feedback = top_with_full_feedback[0].node1.attributes["name"]

    assert winner_without_feedback == winner_with_full_feedback, (
        "Edge-only retrieval ranking flipped between feedback_influence=0.0 "
        "and feedback_influence=1.0. The high-node-feedback pair overtook the "
        "low-node-feedback pair, which means the penalty placeholder on the "
        "missing nodes was being blended by feedback instead of held fixed."
    )
