from typing import Type, List
from pydantic import BaseModel, create_model
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.engine.models.Entity import Entity


async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = render_prompt("generate_graph_prompt.txt", {})
    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph


class EntityListResponse(BaseModel):
    """Response model for entity extraction containing a list of entities."""

    entities: List[Entity]


class PotentialNodesAndRelationships(BaseModel):
    """Response model containing lists of potential node names and relationships."""

    nodes: List[str]
    relationships: List[str]


async def extract_content_nodes_and_relationships(
    content: str, n_rounds: int = 2
) -> tuple[List[str], List[str]]:
    """Extracts node names and relationships from content through multiple rounds of analysis."""
    llm_client = get_llm_client()
    all_nodes: List[str] = []
    all_relationships: List[str] = []
    existing_nodes = set()  # Track existing node names in lowercase
    existing_relationships = set()  # Track existing relationship names in lowercase

    for round_num in range(n_rounds):
        context = {
            "previous_nodes": all_nodes,
            "previous_relationships": all_relationships,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        system_prompt = render_prompt("extract_graph_relationships_prompt.txt", context)
        response = await llm_client.acreate_structured_output(
            text_input=content,
            system_prompt=system_prompt,
            response_model=PotentialNodesAndRelationships,
        )

        # Only add new nodes that haven't been seen before
        for node in response.nodes:
            if node.lower() not in existing_nodes:
                all_nodes.append(node)
                existing_nodes.add(node.lower())

        # Only add new relationships that haven't been seen before
        for relationship in response.relationships:
            if relationship.lower() not in existing_relationships:
                all_relationships.append(relationship)
                existing_relationships.add(relationship.lower())

    return all_nodes, all_relationships


async def extract_content_edges(
    content: str, potential_nodes: List[str], potential_relationships: List[str]
) -> KnowledgeGraph:
    """Creates a knowledge graph by identifying relationships between the provided potential nodes."""
    llm_client = get_llm_client()

    context = {
        "potential_nodes": potential_nodes,
        "potential_relationships": potential_relationships,
    }

    system_prompt = render_prompt("extract_graph_edges_prompt.txt", context)
    graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model=KnowledgeGraph
    )

    # Create a map of node names to ensure we don't duplicate nodes
    node_map = {}
    for node in graph.nodes:
        if node.name not in node_map:
            node_map[node.name] = node.id

    # Filter edges to ensure they only connect existing nodes
    valid_edges = []
    for edge in graph.edges:
        source_exists = any(node.id == edge.source_node_id for node in graph.nodes)
        target_exists = any(node.id == edge.target_node_id for node in graph.nodes)
        if source_exists and target_exists:
            valid_edges.append(edge)

    return KnowledgeGraph(nodes=list(graph.nodes), edges=valid_edges)
