from pathlib import Path

from cognee.shared.data_models import Edge as KGEdge


def test_kg_edge_accepts_missing_description():
    edge = KGEdge(source_node_id="Alice", target_node_id="Acme", relationship_name="works_at")

    assert edge.description is None
    assert "description" in edge.model_dump()


def test_kg_edge_preserves_description():
    edge = KGEdge(
        source_node_id="Alice",
        target_node_id="Acme",
        relationship_name="works_at",
        description="Alice works at Acme.",
    )

    assert edge.description == "Alice works at Acme."
    assert edge.model_dump()["description"] == "Alice works at Acme."


def test_generate_graph_prompt_requests_concrete_edge_descriptions():
    prompt_path = Path(__file__).parents[3] / "infrastructure/llm/prompts/generate_graph_prompt.txt"
    prompt = prompt_path.read_text()

    assert "Every edge should include a description" in prompt
    assert "stay dry and efficient" in prompt
    assert "Alice works at Acme as a platform engineer on the search team." in prompt
    assert "Do not add outside knowledge." in prompt
    assert "This edge describes an employment relationship." in prompt
