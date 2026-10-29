from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.fact_conflicts import is_rejected_fact, user_selected_facts
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.utils.conflict_context import (
    attach_chunk_conflicts,
    render_chunk_context,
)
from cognee.modules.retrieval.utils.resolved_context import (
    apply_entity_resolutions,
    apply_fact_hit_resolutions,
    apply_graph_resolutions,
    read_user_resolutions,
    user_resolution_text,
)


@pytest.mark.asyncio
async def test_chunks_result_retains_unrelated_facts_and_annotates_the_user_decision():
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(return_value=([], [])),
        get_neighborhood=AsyncMock(
            return_value=(
                [
                    (
                        "conflict",
                        {
                            "resolution_source": "user",
                            "selected_fact": "Alice is CEO",
                            "text": "Resolved: Alice is CEO",
                        },
                    )
                ],
                [("conflict", "chunk", "conflict_cites", {"document": "Report"})],
            )
        ),
    )
    passage = "Bob is CEO. The company was founded in 1998."
    hit = SimpleNamespace(id="chunk", payload={"id": "chunk", "text": passage}, score=0.1)
    unified = SimpleNamespace(
        graph=graph, vector=SimpleNamespace(search=AsyncMock(return_value=[hit]))
    )
    retriever = ChunksRetriever()
    with patch(
        "cognee.modules.retrieval.chunks_retriever.get_unified_engine",
        new=AsyncMock(return_value=unified),
    ):
        objects = await retriever.get_retrieved_objects("When was the company founded?")
    context = await retriever.get_context_from_objects("When was it founded?", objects)
    result = await retriever.get_completion_from_context("When was it founded?", objects, context)
    assert result[0]["text"] == passage
    assert result[0]["resolved_facts"] == {"conflict": "Alice is CEO"}
    assert result[0]["conflicts"] == [user_resolution_text("Alice is CEO")]
    assert passage in context
    assert user_resolution_text("Alice is CEO") in context


def test_standalone_vector_fact_returns_the_selected_information_without_mutating_store():
    old = "Company has CEO Bob"
    hit = SimpleNamespace(id=str(EdgeType.id_for(old)), payload={"relationship_name": old})
    resolutions = [{"selected_fact": "Company has CEO Alice", "rejected_fact_texts": [old]}]
    result = apply_fact_hit_resolutions([hit], resolutions)
    assert result[0].payload["relationship_name"] == "Company has CEO Alice"
    assert hit.payload["relationship_name"] == old


@pytest.mark.asyncio
async def test_graph_retrieval_replaces_only_rejected_edges():
    source = Node("company", {"name": "Company"})
    target = Node("bob", {"name": "Bob", "description": "Bob is CEO"})
    edge = Edge(source, target, attributes={"edge_object_id": "old", "edge_text": "Bob is CEO"})
    resolutions = [
        {
            "selected_fact": "Alice is CEO",
            "rejected_fact_ids": ["old"],
            "sources": [],
            "id": "conflict",
            "about_id": "company",
            "selected_fact_id": "chosen",
        }
    ]
    result = apply_graph_resolutions([edge], resolutions)
    text = await resolve_edges_to_text(result)
    assert "Alice is CEO" in text and "Bob" not in text
    assert edge.node2.attributes["name"] == "Bob"
    assert edge.attributes["edge_text"] == "Bob is CEO"


@pytest.mark.asyncio
async def test_resolution_reads_are_scoped_and_decode_json_aliases():
    dataset = uuid4()
    props = {
        "resolution_source": "user",
        "selected_fact": "Alice is CEO",
        "dataset_id": str(dataset),
        "sources_json": '["chunk"]',
        "rejected_fact_ids_json": '["old"]',
    }
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(
            return_value=(
                [
                    ("own", props),
                    ("other", {**props, "dataset_id": str(uuid4()), "selected_fact": "private"}),
                ],
                [],
            )
        )
    )
    token = current_dataset_id.set(dataset)
    try:
        result = await read_user_resolutions(graph)
    finally:
        current_dataset_id.reset(token)
    assert len(result) == 1
    assert result[0]["sources"] == ["chunk"] and result[0]["rejected_fact_ids"] == ["old"]


@pytest.mark.asyncio
async def test_pending_decision_annotates_summary_even_without_citation_links():
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(
            return_value=(
                [
                    (
                        "conflict",
                        {
                            "resolution_source": "user",
                            "selected_fact": "Alice is CEO",
                            "sources": ["chunk"],
                            "write_pending": True,
                        },
                    )
                ],
                [],
            )
        ),
        get_neighborhood=AsyncMock(return_value=([], [])),
    )
    text = "Bob is CEO. Revenue grew 20 percent."
    hits = [{"id": "summary", "source_chunk_id": "chunk", "text": text}]
    await attach_chunk_conflicts(hits, graph_engine=graph, summaries=True)
    assert hits[0]["text"] == text
    assert hits[0]["resolved_facts"] == {"conflict": "Alice is CEO"}
    assert user_resolution_text("Alice is CEO") in render_chunk_context(hits)


@pytest.mark.asyncio
async def test_failure_reading_user_decisions_does_not_return_rejected_source():
    graph = SimpleNamespace(get_filtered_graph_data=AsyncMock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        await attach_chunk_conflicts([{"id": "chunk", "text": "Bob is CEO"}], graph_engine=graph)


def test_entity_with_rejected_bullet_keeps_selected_fact_outside_top_k():
    entities = [
        {
            "id": "company",
            "edges": [{"edge_object_id": "old", "text": "Bob is CEO"}],
            "conflicts": ["Alice is CEO", "Resolved by user: Alice is CEO"],
        }
    ]
    apply_entity_resolutions(
        entities,
        [
            {
                "about_id": "company",
                "selected_fact": "Alice is CEO",
                "rejected_fact_ids": ["old"],
                "text": "Resolved by user: Alice is CEO",
            }
        ],
    )
    assert entities[0]["edges"] == []
    assert entities[0]["conflicts"] == [user_resolution_text("Alice is CEO")]


@pytest.mark.asyncio
async def test_resolution_preserves_shared_entities_source_text_and_unrelated_graph_edges():
    paris = Node("paris", {"name": "Paris", "description": "Paris is a city in France."})
    company = Node("company", {"name": "Acme", "description": "Acme manufactures widgets."})
    chunk = Node("chunk", {"text": "Acme HQ is in London. Acme was founded in 1998."})
    edges = [
        Edge(
            paris,
            Node("france", {"name": "France"}),
            attributes={
                "relationship_name": "capital_of",
                "edge_text": "Paris is the capital of France.",
            },
        ),
        Edge(
            chunk,
            company,
            attributes={"relationship_name": "contains", "edge_text": "The report describes Acme."},
        ),
    ]
    result = apply_graph_resolutions(
        edges,
        [
            {
                "about_id": "company",
                "values": ["paris", "london"],
                "selected_fact": "Acme HQ is in Paris",
                "sources": ["chunk"],
                "rejected_fact_ids": ["old-headquarters-fact"],
            }
        ],
    )
    for original, updated in zip(edges, result):
        assert updated.attributes == original.attributes
        assert updated.node1.attributes == original.node1.attributes
        assert updated.node2.attributes == original.node2.attributes
    text = await resolve_edges_to_text(result)
    assert "Paris is a city in France." in text
    assert "Paris is the capital of France." in text
    assert "Acme was founded in 1998." in text
    assert "Acme manufactures widgets." in text
    assert "Acme HQ is in Paris" not in text


def test_entity_resolution_preserves_unrelated_source_bullets():
    bullet = {
        "source_id": "chunk",
        "target_id": "company",
        "edge_object_id": "contains",
        "text": "Acme was founded in 1998.",
    }
    entities = [{"id": "company", "description": "Acme manufactures widgets.", "edges": [bullet]}]
    apply_entity_resolutions(
        entities,
        [
            {
                "about_id": "company",
                "selected_fact": "Acme HQ is in Paris",
                "sources": ["chunk"],
                "rejected_fact_ids": ["old-headquarters-fact"],
            }
        ],
    )
    assert entities[0]["description"] == "Acme manufactures widgets."
    assert entities[0]["edges"][0]["text"] == "Acme was founded in 1998."
    assert entities[0]["conflicts"] == [user_resolution_text("Acme HQ is in Paris")]


def test_manual_edge_marks_are_scoped_to_the_current_dataset():
    dataset = uuid4()
    props = {
        "conflict_marks": [
            {
                "dataset_id": str(dataset),
                "resolution_source": "user",
                "status": "superseded",
                "selected_fact": "Alice is CEO",
            }
        ]
    }
    token = current_dataset_id.set(uuid4())
    try:
        assert not is_rejected_fact(props)
        assert user_selected_facts(props) == []
        current_dataset_id.set(dataset)
        assert is_rejected_fact(props)
        assert user_selected_facts(props) == ["Alice is CEO"]
    finally:
        current_dataset_id.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retriever_type", [GraphCompletionCotRetriever, GraphCompletionContextExtensionRetriever]
)
async def test_followup_retrieval_applies_saved_decision(retriever_type):
    edge = Edge(
        Node("company"),
        Node("bob"),
        attributes={"edge_object_id": "old", "edge_text": "Bob is CEO"},
    )
    retriever = retriever_type()
    retriever.get_triplets = AsyncMock(return_value=[edge])
    retriever._unified_engine = SimpleNamespace(
        graph=SimpleNamespace(
            get_filtered_graph_data=AsyncMock(
                return_value=(
                    [
                        (
                            "conflict",
                            {
                                "resolution_source": "user",
                                "selected_fact": "Alice is CEO",
                                "about_id": "company",
                                "selected_fact_id": "chosen",
                                "rejected_fact_ids": ["old"],
                            },
                        )
                    ],
                    [],
                )
            )
        )
    )
    batches = await retriever.get_triplets_batch(["Who is CEO?"])
    text = await resolve_edges_to_text(batches[0])
    assert "Alice is CEO" in text and "Bob is CEO" not in text


@pytest.mark.asyncio
async def test_a_retrieved_selected_edge_states_the_decision_once():
    company = Node("company", {"name": "Company"})
    decision = {
        "id": "conflict",
        "about_id": "company",
        "selected_fact": "Company has CEO Alice",
        "selected_fact_id": "alice-fact",
        "rejected_fact_ids": ["bob-fact"],
        "sources": [],
        "values": [],
    }
    mark = {
        "conflict_id": "conflict",
        "resolution_source": "user",
        "selected_fact": "Company has CEO Alice",
    }
    alice = Edge(
        company,
        Node("alice", {"name": "Alice"}),
        attributes={
            "edge_object_id": "alice-fact",
            "edge_text": "Company has CEO Alice",
            "conflict_marks": [{**mark, "status": "current"}],
        },
    )
    bob = Edge(
        company,
        Node("bob", {"name": "Bob"}),
        attributes={
            "edge_object_id": "bob-fact",
            "edge_text": "Company has CEO Bob",
            "conflict_marks": [{**mark, "status": "superseded"}],
        },
    )
    carol = Edge(
        company,
        Node("carol", {"name": "Carol"}),
        attributes={
            "edge_object_id": "carol-fact",
            "edge_text": "Company has CEO Carol",
            "conflict_marks": [{**mark, "status": "superseded"}],
        },
    )
    decision["rejected_fact_ids"].append("carol-fact")
    # The retrieved selected edge states the decision: the rejected edges are dropped
    # instead of being restated as a synthetic link or an accepted-facts section.
    result = apply_graph_resolutions([alice, bob, carol], [decision])
    assert [edge.attributes["edge_object_id"] for edge in result] == ["alice-fact"]
    text = await resolve_edges_to_text(result)
    assert "Company has CEO Bob" not in text and "Company has CEO Carol" not in text
    assert "accepted_fact" not in text and "## Accepted facts" not in text
    assert text.count("Company has CEO Alice") == 1
    assert "--> Alice [accepted by user]" in text
    # Without it, one synthetic link states the decision, however many claims it rejected.
    text = await resolve_edges_to_text(apply_graph_resolutions([bob, carol], [decision]))
    assert "Company has CEO Bob" not in text and "Company has CEO Carol" not in text
    assert text.count("--[accepted_fact]-->") == 1 and "## Accepted facts" not in text
    # Edges rendered from their marks alone: the section restates only an absent choice.
    text = await resolve_edges_to_text([alice, bob])
    assert "Company has CEO Bob" not in text and "## Accepted facts" not in text
    assert text.count("Company has CEO Alice") == 1
    text = await resolve_edges_to_text([bob, carol])
    assert text == "## Accepted facts (user resolved)\nCompany has CEO Alice"


@pytest.mark.asyncio
async def test_the_decision_node_labels_the_selected_edge_next_to_later_evidence():
    """A rebuilt document re-creates the selected edge without its marks; the decision
    node still labels it, ahead of another conflict's review status, so graph context
    keeps the choice next to evidence a review found later."""
    company = Node("company", {"name": "Company"})
    decision = {
        "id": "conflict",
        "about_id": "company",
        "dataset_id": "dataset",
        "selected_fact": "Company has CEO Alice",
        "selected_fact_id": "alice-fact",
        "rejected_fact_ids": ["bob-fact"],
        "sources": [],
        "values": [],
    }
    alice = Edge(
        company,
        Node("alice", {"name": "Alice"}),
        attributes={
            "edge_object_id": "alice-fact",
            "edge_text": "Company has CEO Alice",
            "effective_date": "2024-01-01",
            "conflict_marks": [{"conflict_id": "employer", "status": "conflicting"}],
        },
    )
    bob = Edge(
        company,
        Node("bob", {"name": "Bob"}),
        attributes={"edge_object_id": "bob-fact", "edge_text": "Company has CEO Bob"},
    )
    carol = Edge(
        company,
        Node("carol", {"name": "Carol"}),
        attributes={
            "edge_object_id": "carol-fact",
            "edge_text": "Company has CEO Carol",
            "effective_date": "2026-06-01",
            "conflict_marks": [{"conflict_id": "conflict", "status": "current"}],
        },
    )
    token = current_dataset_id.set("dataset")
    try:
        result = apply_graph_resolutions([alice, bob, carol], [decision])
        text = await resolve_edges_to_text(result)
    finally:
        current_dataset_id.reset(token)
    assert "--[Company has CEO Alice]--> Alice [accepted by user; as of 2024-01-01]" in text
    assert "--[Company has CEO Carol]--> Carol [as of 2026-06-01]" in text
    assert "Company has CEO Bob" not in text and "accepted_fact" not in text
    assert alice.attributes["conflict_marks"] == [
        {"conflict_id": "employer", "status": "conflicting"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_first", [True, False])
async def test_a_synthetic_link_keeps_the_subjects_name_on_its_other_edges(subject_first):
    company = Node("company", {"name": "Company", "description": "Company makes widgets."})
    bob = Node("bob", {"name": "Bob"})
    decision = {
        "id": "conflict",
        "about_id": "company",
        "selected_fact": "Company has CEO Alice",
        "selected_fact_id": "alice-fact",
        "rejected_fact_ids": ["bob-fact"],
        "sources": [],
        "values": [],
    }
    rejected = Edge(
        *((company, bob) if subject_first else (bob, company)),
        attributes={"edge_object_id": "bob-fact", "edge_text": "Company has CEO Bob"},
    )
    office = Edge(
        company,
        Node("paris", {"name": "Paris"}),
        attributes={"edge_object_id": "office", "relationship_name": "headquartered_in"},
    )
    text = await resolve_edges_to_text(apply_graph_resolutions([rejected, office], [decision]))
    assert "--[accepted_fact]--> Company  (Company has CEO Alice)" in text
    assert "Company --[headquartered_in]--> Paris" in text
    assert "Company makes widgets." in text and "Bob" not in text
