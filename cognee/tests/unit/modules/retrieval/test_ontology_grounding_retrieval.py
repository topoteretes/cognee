"""Ontology grounding wired into retrieval: pinned seeds and the context block."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from rdflib import OWL, RDF, RDFS, Graph, Namespace

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology.query_grounding import (
    GroundedConcept,
    ground_query,
    with_grounding_block,
)
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid.context import format_hybrid_context
from cognee.modules.retrieval.hybrid.entities import build_entities, pin_grounded_entities
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


def _grounding() -> list[GroundedConcept]:
    return [
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
    ]


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


def test_pin_node_ids_creates_a_missing_collection():
    search = NodeEdgeVectorSearch(vector_engine=MagicMock())
    search.query_list_length = None
    search.node_distances = {}

    search.pin_node_ids({"EntityType_name": [EXPOSURE_ID]})

    assert [str(r.id) for r in search.node_distances["EntityType_name"]] == [EXPOSURE_ID]


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
    vector_engine.existing_collections = AsyncMock(side_effect=lambda names: list(names))

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


# --- GraphCompletionRetriever -----------------------------------------------------------


@pytest.fixture
def ontology(monkeypatch):
    """Both retrievers ground against the test ontology, as with ONTOLOGY_FILE_PATH set."""
    resolver = _resolver()

    async def ground(query):
        return ground_query(query, resolver)

    for module in (
        "cognee.modules.ontology.query_grounding",
        "cognee.modules.retrieval.graph_completion_retriever",
        "cognee.modules.retrieval.hybrid_retriever",
    ):
        monkeypatch.setattr(f"{module}.ground", ground)


@pytest.mark.asyncio
async def test_graph_completion_pins_grounded_nodes_as_seeds(ontology):
    retriever = GraphCompletionRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await retriever.get_triplets("What is our credit exposure to Acme?")

    pinned = mock_search.call_args.kwargs["pinned_node_ids"]
    assert pinned == {"EntityType_name": [EXPOSURE_ID], "Entity_name": [ACME_ID]}


@pytest.mark.asyncio
async def test_graph_completion_passes_no_pins_when_grounding_misses(ontology):
    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await GraphCompletionRetriever().get_triplets("hello there")
        assert mock_search.call_args.kwargs["pinned_node_ids"] is None

        # Batch mode never pins: the scorer's distances are per-query lists there.
        await GraphCompletionRetriever().get_triplets(query_batch=["credit exposure", "Acme"])
        assert mock_search.call_args.kwargs["pinned_node_ids"] is None


@pytest.mark.asyncio
async def test_neighbourhood_mode_pins_no_seeds_but_keeps_the_block(ontology):
    # Neighbourhood seeds are the first vector hits, and that projection skips node_name.
    retriever = GraphCompletionRetriever(node_name=["project_a"], neighborhood_depth=1)

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await retriever.get_triplets("What is our credit exposure to Acme?")
    assert mock_search.call_args.kwargs["pinned_node_ids"] is None

    with patch.object(
        GraphCompletionRetriever, "resolve_edges_to_text", new_callable=AsyncMock, return_value="x"
    ):
        context = await retriever.get_context_from_objects(
            query="credit exposure of Acme", retrieved_objects=[MagicMock()]
        )
    assert context.startswith("## Ontology grounding\n")


@pytest.mark.asyncio
async def test_graph_completion_prepends_grounding_block_only_with_real_context(ontology):
    retriever = GraphCompletionRetriever()
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


@pytest.mark.asyncio
async def test_chain_of_thought_context_carries_the_grounding_block(ontology):
    from types import SimpleNamespace

    from cognee.modules.retrieval.graph_completion_cot_retriever import (
        GraphCompletionCotRetriever,
    )

    state = SimpleNamespace(triplets=None, context_text="")
    with (
        patch.object(
            GraphCompletionCotRetriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[MagicMock()]],
        ),
        patch.object(
            GraphCompletionCotRetriever,
            "resolve_edges_to_text",
            new_callable=AsyncMock,
            return_value="Node1: CreditExposure -- measured_for -- Acme",
        ),
    ):
        await GraphCompletionCotRetriever()._fetch_initial_triplets_and_context(
            {"credit exposure of Acme": state}
        )

    assert state.context_text.startswith("## Ontology grounding\n")
    assert state.context_text.endswith("Node1: CreditExposure -- measured_for -- Acme")


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
    assert pin_grounded_entities([other], []) == [other]


@pytest.mark.asyncio
async def test_hybrid_context_leads_with_the_grounding_block_only_with_content(ontology):
    retriever = HybridRetriever()
    passages = {"chunks": [{"text": "Acme owes 3M."}]}

    with patch.object(
        HybridRetriever,
        "_build_global_context_section",
        new_callable=AsyncMock,
        return_value="## Global context\nroot",
    ):
        context = await retriever.get_context_from_objects(
            query="credit exposure of Acme", retrieved_objects=passages
        )
        batch = await retriever.get_context_from_objects(
            query_batch=["credit exposure", "hello"], retrieved_objects=[passages, passages]
        )

    sections = context.split("\n\n")
    assert sections[0].startswith("## Ontology grounding\n")
    assert sections[1].startswith("## Global context")
    assert "Acme owes 3M." in context
    assert batch[0].startswith("## Ontology grounding\n")
    assert batch[1].startswith("## Global context")

    # Nothing retrieved -> no context at all, grounding or not.
    assert await with_grounding_block("credit exposure of Acme", "") == ""


def _graph_engine(nodes: list, edges: list) -> MagicMock:
    graph_engine = MagicMock()
    graph_engine.get_neighborhood = AsyncMock(return_value=(nodes, edges))
    return graph_engine


PINNED_IDS = {EXPOSURE_ID, ACME_ID}


@pytest.mark.asyncio
async def test_a_pinned_concept_the_graph_does_not_hold_is_not_shown():
    hits = pin_grounded_entities([], _grounding())
    entities, _ = await build_entities(_graph_engine([], []), hits, 10, pinned_ids=PINNED_IDS)

    assert entities == []
    assert format_hybrid_context("", {"entities": entities}) == ""

    # Dropped before the entities_top_k cut, so it takes no vector hit's slot.
    vector_ids = [str(uuid4()) for _ in range(5)]
    vector_hits = [
        ScoredResult(id=UUID(node_id), score=0.3, payload={"name": f"entity {index}"})
        for index, node_id in enumerate(vector_ids)
    ]
    hits = pin_grounded_entities(vector_hits, _grounding())
    entities, _ = await build_entities(_graph_engine([], []), hits, 10, pinned_ids=PINNED_IDS)

    assert [entity["id"] for entity in entities] == vector_ids


@pytest.mark.asyncio
async def test_a_pinned_concept_shows_only_neighbours_in_the_node_set():
    customer_id = str(EntityType.id_for("Customer"))
    initech_id = str(Entity.id_for("Initech"))
    acme_id, globex_id = str(uuid4()), str(uuid4())
    nodes = [
        (customer_id, {"name": "customer"}),
        (initech_id, {"name": "Initech", "belongs_to_set": ["project_b"]}),
        (acme_id, {"name": "Acme", "belongs_to_set": ["project_a"]}),
        (globex_id, {"name": "Globex", "belongs_to_set": ["project_b"]}),
    ]
    edges = [
        (acme_id, customer_id, "is_a", {}),
        (globex_id, customer_id, "is_a", {}),
        (initech_id, customer_id, "is_a", {}),
        (initech_id, globex_id, "supplies", {}),
    ]
    grounding = [
        GroundedConcept("customers", "Customer", "classes", customer_id),
        GroundedConcept("initech", "Initech", "individuals", initech_id),
    ]

    entities, _ = await build_entities(
        _graph_engine(nodes, edges),
        pin_grounded_entities([], grounding),
        10,
        node_name=["project_a"],
        pinned_ids={customer_id, initech_id},
    )

    # Initech's neighbours are all outside project_a, its is_a edge included.
    assert [entity["name"] for entity in entities] == ["Customer"]
    assert [edge["source"] for edge in entities[0]["edges"]] == ["Acme"]


@pytest.mark.asyncio
async def test_a_class_keeps_its_realizes_edge_next_to_many_members():
    customer_id = str(EntityType.id_for("Customer"))
    member_ids = [str(uuid4()) for _ in range(12)]
    nodes = [(customer_id, {"name": "customer"}), ("crm", {"name": "crm.customers"})]
    nodes += [
        (member_id, {"name": f"customer {index}"}) for index, member_id in enumerate(member_ids)
    ]
    edges = [(member_id, customer_id, "is_a", {}) for member_id in member_ids]
    edges.append(("crm", customer_id, "realizes", {}))
    hit = ScoredResult(id=UUID(customer_id), score=0.0, payload={"name": "customer"})

    entities, _ = await build_entities(_graph_engine(nodes, edges), [hit], 10)

    assert "crm.customers -- realizes -- customer" in [
        edge["text"] for edge in entities[0]["edges"]
    ]

    # An ordinary entity's own is_a edge still comes first.
    acme_id = member_ids[0]
    hit = ScoredResult(id=UUID(acme_id), score=0.3, payload={"name": "customer 0"})
    acme_edges = [("crm", acme_id, "mentions", {}), (acme_id, customer_id, "is_a", {})]
    entities, _ = await build_entities(_graph_engine(nodes, acme_edges), [hit], 10)

    assert entities[0]["edges"][0]["text"] == "customer 0 -- is_a -- customer"


async def _retrieve_hybrid(query: str):
    """Run ``HybridRetriever._retrieve_one`` with every store mocked out.

    Returns the result and the ``build_entities`` mock, whose second argument holds
    the entity hits after pinning.
    """
    retriever = HybridRetriever()
    retriever._unified_engine = MagicMock()
    retriever._unified_engine.vector.embedding_engine.embed_text = AsyncMock(
        return_value=[[0.1, 0.2]]
    )
    # Dev reads user conflict resolutions off the graph (SDK-1333); none here.
    retriever._unified_engine.graph.get_filtered_graph_data = AsyncMock(return_value=([], []))
    truth = MagicMock(q_coords=None, truth_state_by_id={}, current_truth_epoch=None)
    module = "cognee.modules.retrieval.hybrid_retriever"

    with (
        patch(f"{module}.build_truth_context", new_callable=AsyncMock, return_value=truth),
        patch(f"{module}.load_preference_weights", new_callable=AsyncMock, return_value={}),
        patch(f"{module}.dlt_rows_indexed", new_callable=AsyncMock, return_value=False),
        patch(f"{module}.attach_chunk_conflicts", new_callable=AsyncMock),
        patch(
            f"{module}.retrieve_hybrid_chunks",
            new_callable=AsyncMock,
            return_value={"chunks": [], "chunk_summaries": {}},
        ),
        patch(f"{module}.search_entities", new_callable=AsyncMock, return_value=[]),
        patch(f"{module}.search_collection", new_callable=AsyncMock, return_value=[]),
        patch(
            f"{module}.build_entities", new_callable=AsyncMock, return_value=([], set())
        ) as mock_build_entities,
    ):
        result = await retriever._retrieve_one(query)
    return result, mock_build_entities


@pytest.mark.asyncio
async def test_hybrid_pins_the_grounded_concepts_in_the_entity_lane(ontology):
    _, mock_build_entities = await _retrieve_hybrid("What is our credit exposure to Acme?")

    pinned_hits = mock_build_entities.call_args.args[1]
    assert [str(hit.id) for hit in pinned_hits] == [EXPOSURE_ID, ACME_ID]

    _, mock_build_entities = await _retrieve_hybrid("hello there")
    assert mock_build_entities.call_args.args[1] == []


# --- ONTOLOGY_QUERY_GROUNDING=false -----------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_switched_off_pins_nothing_and_renders_no_block(monkeypatch):
    from cognee.modules.ontology import query_grounding
    from cognee.modules.ontology.ontology_env_config import OntologyEnvConfig

    config = OntologyEnvConfig(ontology_file_path="/ontology.owl", ontology_query_grounding=False)
    monkeypatch.setattr(query_grounding, "get_ontology_env_config", lambda: config)

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await GraphCompletionRetriever().get_triplets("credit exposure of Acme")
    assert mock_search.call_args.kwargs["pinned_node_ids"] is None

    with patch.object(
        GraphCompletionRetriever, "resolve_edges_to_text", new_callable=AsyncMock, return_value="x"
    ):
        context = await GraphCompletionRetriever().get_context_from_objects(
            query="credit exposure of Acme", retrieved_objects=[MagicMock()]
        )
    assert context == "x"

    _, mock_build_entities = await _retrieve_hybrid("credit exposure of Acme")
    assert mock_build_entities.call_args.args[1] == []
    assert await with_grounding_block("credit exposure of Acme", "x") == "x"


# --- authoritative sources in the entity lane ---------------------------------------


@pytest.mark.asyncio
async def test_authoritative_realizes_edges_survive_the_edge_cap_and_say_so():
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
    assert bullets[0]["authoritative"] is True
    assert bullets[0]["text"].endswith("(authoritative source, owner: sales-ops)")
    assert bullets[1]["authoritative"] is False
