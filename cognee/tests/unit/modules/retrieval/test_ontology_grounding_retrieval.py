"""Ontology grounding wired into retrieval: pinned seeds and the context block."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology.query_grounding import GroundedConcept, QueryGrounding
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid.context import format_hybrid_context
from cognee.modules.retrieval.hybrid.entities import pin_grounded_entities
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch

NS = Namespace("http://example.org/fin#")


def _resolver() -> RDFLibOntologyResolver:
    graph = Graph()
    for class_name in ("RiskMetric", "CreditExposure", "Customer"):
        graph.add((NS[class_name], RDF.type, OWL.Class))
    graph.add((NS.CreditExposure, RDFS.subClassOf, NS.RiskMetric))
    graph.add((NS.Acme, RDF.type, NS.Customer))
    resolver = RDFLibOntologyResolver(ontology_file=None)
    resolver.graph = graph
    resolver.build_lookup()
    return resolver


EXPOSURE_ID = str(EntityType.id_for("CreditExposure"))
ACME_ID = str(Entity.id_for("Acme"))


def _grounding() -> QueryGrounding:
    return QueryGrounding(
        query="credit exposure of Acme",
        concepts=(
            GroundedConcept(
                term="credit exposure",
                canonical_name="CreditExposure",
                category="classes",
                node_id=EXPOSURE_ID,
                parents=("RiskMetric",),
            ),
            GroundedConcept(
                term="acme",
                canonical_name="Acme",
                category="individuals",
                node_id=ACME_ID,
                parents=("Customer",),
            ),
        ),
    )


# --- NodeEdgeVectorSearch.pin_node_ids -------------------------------------------------


def test_pin_node_ids_adds_exact_hits_and_rescored_duplicates_win():
    search = NodeEdgeVectorSearch(vector_engine=MagicMock())
    search.query_list_length = None
    other_id = uuid4()
    search.node_distances = {
        "Entity_name": [ScoredResult(id=other_id, score=0.4, payload={})],
        "EntityType_name": [ScoredResult(id=UUID(EXPOSURE_ID), score=0.9, payload={})],
    }

    search.pin_node_ids({"EntityType_name": [EXPOSURE_ID], "Entity_name": [ACME_ID]})

    assert [r.score for r in search.node_distances["EntityType_name"]] == [0.9, 0.0]
    assert str(search.node_distances["EntityType_name"][-1].id) == EXPOSURE_ID
    assert {str(r.id) for r in search.node_distances["Entity_name"]} == {str(other_id), ACME_ID}
    assert set(search.extract_relevant_node_ids()) == {str(other_id), EXPOSURE_ID, ACME_ID}
    assert search.has_results()


def test_pin_node_ids_creates_missing_collection_and_ignores_batch_mode_and_edges():
    search = NodeEdgeVectorSearch(vector_engine=MagicMock())
    search.query_list_length = None
    search.node_distances = {}

    search.pin_node_ids({"EntityType_name": [EXPOSURE_ID], "EdgeType_relationship_name": [ACME_ID]})
    assert list(search.node_distances) == ["EntityType_name"]
    assert search.edge_distances == []

    batch = NodeEdgeVectorSearch(vector_engine=MagicMock())
    batch.query_list_length = 2
    batch.node_distances = {}
    batch.pin_node_ids({"EntityType_name": [EXPOSURE_ID]})
    assert batch.node_distances == {}


@pytest.mark.asyncio
async def test_pinned_ids_flow_into_projection_and_outrank_vector_hits():
    """End to end through brute_force_triplet_search with mocked engines: the pinned
    ontology node is projected even though the vector search never returned it, and
    the triplet touching it ranks first."""
    from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
    from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
    from cognee.modules.retrieval.utils.brute_force_triplet_search import (
        brute_force_triplet_search,
    )

    vector_hit_id = str(uuid4())
    far_id = str(uuid4())
    graph = CogneeGraph()
    exposure = Node(EXPOSURE_ID, {"name": "CreditExposure"})
    hit = Node(vector_hit_id, {"name": "some entity"})
    far = Node(far_id, {"name": "unrelated"})
    for node in (exposure, hit, far):
        graph.add_node(node)
    pinned_edge = Edge(exposure, far, attributes={"relationship_name": "measured_for"})
    hit_edge = Edge(hit, far, attributes={"relationship_name": "mentions"})
    graph.add_edge(pinned_edge)
    graph.add_edge(hit_edge)

    vector_engine = AsyncMock()
    vector_engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2]])

    async def search(collection_name, **_kwargs):
        if collection_name == "Entity_name":
            return [ScoredResult(id=UUID(vector_hit_id), score=0.3, payload={})]
        return []

    vector_engine.search = AsyncMock(side_effect=search)

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            new_callable=AsyncMock,
            return_value=graph,
        ) as mock_fragment,
    ):
        results = await brute_force_triplet_search(
            query="credit exposure",
            top_k=2,
            pinned_node_ids={"EntityType_name": [EXPOSURE_ID]},
        )

    projected_ids = set(mock_fragment.call_args.kwargs["relevant_ids_to_filter"])
    assert {EXPOSURE_ID, vector_hit_id} <= projected_ids
    assert exposure.attributes["vector_distance"] == [0.0]
    assert hit.attributes["vector_distance"] == [0.3]
    assert results[0] is pinned_edge


def test_pin_node_ids_skips_ids_that_are_not_uuids():
    search = NodeEdgeVectorSearch(vector_engine=MagicMock())
    search.query_list_length = None
    search.node_distances = {}
    search.pin_node_ids({"EntityType_name": ["not-a-uuid", EXPOSURE_ID]})
    assert [str(r.id) for r in search.node_distances["EntityType_name"]] == [EXPOSURE_ID]


# --- GraphCompletionRetriever -----------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_completion_pins_grounded_nodes_as_seeds():
    retriever = GraphCompletionRetriever(top_k=5, ontology_resolver=_resolver())

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await retriever.get_triplets("What is our credit exposure to Acme?")

    pinned = mock_search.call_args.kwargs["pinned_node_ids"]
    assert pinned == {"EntityType_name": [EXPOSURE_ID], "Entity_name": [ACME_ID]}


@pytest.mark.asyncio
async def test_graph_completion_passes_no_pins_when_grounding_is_off_or_misses():
    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await GraphCompletionRetriever(
            ontology_resolver=_resolver(), ontology_grounding=False
        ).get_triplets("credit exposure")
        assert mock_search.call_args.kwargs["pinned_node_ids"] is None

        await GraphCompletionRetriever(ontology_resolver=_resolver()).get_triplets("hello there")
        assert mock_search.call_args.kwargs["pinned_node_ids"] is None

        # Batch mode never pins: the scorer's distances are per-query lists there.
        await GraphCompletionRetriever(ontology_resolver=_resolver()).get_triplets(
            query_batch=["credit exposure", "Acme"]
        )
        assert mock_search.call_args.kwargs["pinned_node_ids"] is None


@pytest.mark.asyncio
async def test_graph_completion_prepends_grounding_block_only_with_real_context():
    retriever = GraphCompletionRetriever(ontology_resolver=_resolver())
    edge = MagicMock()

    with patch.object(
        GraphCompletionRetriever,
        "resolve_edges_to_text",
        new_callable=AsyncMock,
        return_value="Node1: CreditExposure -- measured_for -- Acme",
    ):
        context = await retriever.get_context_from_objects(
            query="credit exposure of Acme", retrieved_objects=[edge]
        )

    assert context.startswith("## Ontology grounding\n")
    assert '"credit exposure" refers to CreditExposure (class); is a RiskMetric' in context
    assert context.endswith("Node1: CreditExposure -- measured_for -- Acme")

    # No triplets -> no context at all, grounding or not (keeps "nothing found" detectable).
    empty = await retriever.get_context_from_objects(
        query="credit exposure of Acme", retrieved_objects=[]
    )
    assert empty == ""


# --- Hybrid ---------------------------------------------------------------------------


def test_pin_grounded_entities_leads_with_concepts_and_dedupes():
    existing = ScoredResult(id=UUID(ACME_ID), score=0.3, payload={"id": ACME_ID, "name": "acme"})
    other = ScoredResult(id=uuid4(), score=0.5, payload={"name": "other"})

    hits = pin_grounded_entities([other, existing], _grounding())

    assert [str(hit.id) for hit in hits] == [EXPOSURE_ID, ACME_ID, str(other.id)]
    assert hits[0].score == 0.0
    assert hits[0].payload["name"] == "CreditExposure"
    assert hits[0].payload["type"] == "ontology class"
    assert hits[1].payload["type"] == "Customer"  # an individual is typed by its class
    assert pin_grounded_entities([other], QueryGrounding(query="x")) == [other]


def test_format_hybrid_context_places_grounding_after_global_context_only_with_content():
    grounding_block = _grounding().to_context_block()

    with_passages = format_hybrid_context(
        "",
        {"chunks": [{"text": "Acme owes 3M."}], "ontology_grounding": grounding_block},
    )
    assert with_passages.startswith("## Ontology grounding")
    assert "## Relevant passages" in with_passages

    with_global = format_hybrid_context(
        "## Global context\nroot",
        {"chunks": [{"text": "Acme owes 3M."}], "ontology_grounding": grounding_block},
    )
    sections = with_global.split("\n\n")
    assert sections[0].startswith("## Global context")
    assert sections[1].startswith("## Ontology grounding")

    assert format_hybrid_context("", {"ontology_grounding": grounding_block}) == ""


@pytest.mark.asyncio
async def test_hybrid_retrieve_one_carries_grounding_and_pins_entity_lane():
    retriever = HybridRetriever(ontology_resolver=_resolver())
    retriever._unified_engine = MagicMock()
    retriever._unified_engine.vector.embedding_engine.embed_text = AsyncMock(
        return_value=[[0.1, 0.2]]
    )
    truth = MagicMock(q_coords=None, truth_state_by_id={}, current_truth_epoch=None)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.build_truth_context",
            new_callable=AsyncMock,
            return_value=truth,
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.load_preference_weights",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.retrieve_hybrid_chunks",
            new_callable=AsyncMock,
            return_value={"chunks": [], "chunk_summaries": {}},
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.search_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.search_collection",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.build_entities",
            new_callable=AsyncMock,
            return_value=([], set()),
        ) as mock_build_entities,
    ):
        result = await retriever._retrieve_one("What is our credit exposure to Acme?")

    pinned_hits = mock_build_entities.call_args.args[1]
    assert [str(hit.id) for hit in pinned_hits] == [EXPOSURE_ID, ACME_ID]
    assert result["ontology_grounding"].startswith("## Ontology grounding")


@pytest.mark.asyncio
async def test_hybrid_result_has_no_grounding_key_when_nothing_matches():
    retriever = HybridRetriever(ontology_resolver=_resolver())
    retriever._unified_engine = MagicMock()
    retriever._unified_engine.vector.embedding_engine.embed_text = AsyncMock(
        return_value=[[0.1, 0.2]]
    )
    truth = MagicMock(q_coords=None, truth_state_by_id={}, current_truth_epoch=None)

    with (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.build_truth_context",
            new_callable=AsyncMock,
            return_value=truth,
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.load_preference_weights",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.retrieve_hybrid_chunks",
            new_callable=AsyncMock,
            return_value={"chunks": [], "chunk_summaries": {}},
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.search_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.search_collection",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.build_entities",
            new_callable=AsyncMock,
            return_value=([], set()),
        ),
    ):
        result = await retriever._retrieve_one("hello there")

    assert "ontology_grounding" not in result


# --- authoritative sources in the entity lane ---------------------------------------


@pytest.mark.asyncio
async def test_authoritative_realizes_edges_survive_the_edge_cap_and_say_so():
    from cognee.modules.retrieval.hybrid.entities import build_entities

    customer_id = str(EntityType.id_for("Customer"))
    tables = {f"legacy-{index}": f"legacy_{index}.customer" for index in range(4)}
    tables["crm"] = "crm.customers"
    nodes = [(customer_id, {"name": "customer", "type": "EntityType"})] + [
        (node_id, {"name": name, "type": "SchemaTable"}) for node_id, name in tables.items()
    ]
    edges = [
        (
            node_id,
            customer_id,
            "realizes",
            {
                "relationship_name": "realizes",
                "authoritative": node_id == "crm",
                "owner": "sales-ops" if node_id == "crm" else None,
            },
        )
        for node_id in tables
    ]
    graph_engine = MagicMock()
    graph_engine.get_neighborhood = AsyncMock(return_value=(nodes, edges))
    hit = ScoredResult(
        id=UUID(customer_id), score=0.0, payload={"name": "customer", "type": "EntityType"}
    )

    entities, _ = await build_entities(graph_engine, [hit], max_edges_per_entity=2)

    bullets = entities[0]["edges"]
    assert len(bullets) == 2
    assert bullets[0]["source"] == "crm.customers"
    assert bullets[0]["authoritative"] is True and bullets[0]["owner"] == "sales-ops"
    assert bullets[0]["text"].endswith("(authoritative source, owner: sales-ops)")
    assert bullets[1]["authoritative"] is False
