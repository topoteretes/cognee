import asyncio
from typing import List
from pydantic import BaseModel
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.chunking.models.ExtractionChunk import ExtractionChunk


class PotentialNodesAndRelationships(BaseModel):
    """Response model containing lists of potential node names and relationships."""

    nodes: List[str]
    relationships: List[str]


async def extract_content_nodes_and_relationships(
    extraction_chunk: ExtractionChunk, n_rounds: int = 2
) -> tuple[List[str], List[str]]:
    """Extracts node names and relationships from content through multiple rounds of analysis."""
    llm_client = get_llm_client()
    all_nodes: List[str] = extraction_chunk.potential_nodes.copy()
    all_relationships: List[str] = []
    existing_nodes = {node.lower() for node in all_nodes}  # Track existing node names in lowercase
    existing_relationships = set()  # Track existing relationship names in lowercase

    for round_num in range(n_rounds):
        context = {
            "text": extraction_chunk.document_chunk.text,
            "potential_nodes": extraction_chunk.potential_nodes,
            "previous_nodes": all_nodes,
            "previous_relationships": all_relationships,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        text_input = render_prompt("extract_graph_relationships_prompt_input.txt", context)
        system_prompt = render_prompt("extract_graph_relationships_prompt_system.txt", context)
        response = await llm_client.acreate_structured_output(
            text_input=text_input,
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


async def extract_relationships_from_data(
    extraction_chunks: list[ExtractionChunk], n_rounds: int
) -> List[ExtractionChunk]:
    """Extracts and integrates potential nodes and relationships from extraction chunks using multi-round extraction."""
    chunk_results = await asyncio.gather(
        *[extract_content_nodes_and_relationships(chunk, n_rounds) for chunk in extraction_chunks]
    )

    # Update ExtractionChunks with their relationships (nodes are already present)
    for chunk, (nodes, relationships) in zip(extraction_chunks, chunk_results):
        chunk.potential_relationships = relationships

    return extraction_chunks
