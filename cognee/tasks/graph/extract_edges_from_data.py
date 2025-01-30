import asyncio
from typing import List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.tasks.storage import add_data_points


async def extract_content_edges(chunk: DocumentChunk, n_rounds: int = 2) -> KnowledgeGraph:
    """Creates a knowledge graph by identifying relationships between the provided potential nodes."""
    llm_client = get_llm_client()
    final_graph = KnowledgeGraph(nodes=[], edges=[])
    existing_nodes = set()  # Track existing node names
    existing_edges = set()  # Track existing edges by their source-target-type

    for round_num in range(n_rounds):
        context = {
            "text": chunk.text,
            "potential_nodes": chunk.potential_nodes or [],
            "potential_relationships": chunk.potential_relationships or [],
            "previous_nodes": [node.name for node in final_graph.nodes],
            "previous_edges": [
                (edge.source_node_id, edge.target_node_id, edge.relationship_name)
                for edge in final_graph.edges
            ],
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        text_input = render_prompt("extract_graph_edges_prompt_input.txt", context)
        system_prompt = read_query_prompt("extract_graph_edges_prompt_system.txt")
        round_graph = await llm_client.acreate_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=KnowledgeGraph
        )

        # Add new nodes and edges that haven't been seen before
        for node in round_graph.nodes:
            if node.name not in existing_nodes:
                final_graph.nodes.append(node)
                existing_nodes.add(node.name)

        for edge in round_graph.edges:
            edge_key = (edge.source_node_id, edge.target_node_id, edge.relationship_name)
            if edge_key not in existing_edges:
                source_exists = any(node.id == edge.source_node_id for node in final_graph.nodes)
                target_exists = any(node.id == edge.target_node_id for node in final_graph.nodes)
                if source_exists and target_exists:
                    final_graph.edges.append(edge)
                    existing_edges.add(edge_key)

    return final_graph


async def extract_edges_from_data(
    data_chunks: list[DocumentChunk], n_rounds: int = 2
) -> List[DocumentChunk]:
    """Extracts and integrates edges between nodes using potential nodes and relationships from chunks."""
    chunk_graphs = await asyncio.gather(
        *[extract_content_edges(chunk, n_rounds) for chunk in data_chunks]
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
