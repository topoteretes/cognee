import asyncio
from typing import List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.tasks.storage import add_data_points


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


async def extract_edges_from_data(data_chunks: list[DocumentChunk]) -> List[DocumentChunk]:
    """Extracts and integrates edges between nodes using potential nodes and relationships from chunks."""
    chunk_graphs = await asyncio.gather(
        *[
            extract_content_edges(
                chunk.text, chunk.potential_nodes or [], chunk.potential_relationships or []
            )
            for chunk in data_chunks
        ]
    )
    graph_engine = await get_graph_engine()

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
        graph_engine,
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks,
        chunk_graphs,
        existing_edges_map,
    )

    if len(graph_nodes) > 0:
        await add_data_points(graph_nodes)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return data_chunks
