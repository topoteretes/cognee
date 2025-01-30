import asyncio
from typing import List
from pydantic import BaseModel

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt


class PotentialNodes(BaseModel):
    """Response model containing a list of potential node names."""

    nodes: List[str]


async def extract_content_nodes(content: str, n_rounds: int = 2) -> List[str]:
    """Extracts node names from content through multiple rounds of analysis."""
    llm_client = get_llm_client()
    all_nodes: List[str] = []
    existing_nodes = set()  # Track existing node names in lowercase

    for round_num in range(n_rounds):
        context = {
            "previous_nodes": all_nodes,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        text_input = render_prompt("extract_graph_nodes_prompt_input.txt", context)
        system_prompt = read_query_prompt("extract_graph_nodes_prompt_system.txt")
        response = await llm_client.acreate_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=PotentialNodes
        )

        # Only add new nodes that haven't been seen before
        for node in response.nodes:
            if node.lower() not in existing_nodes:
                all_nodes.append(node)
                existing_nodes.add(node.lower())

    return all_nodes


async def extract_nodes_from_data(
    data_chunks: list[DocumentChunk], n_rounds: int
) -> List[DocumentChunk]:
    """Extracts and integrates potential nodes from document chunks using multi-round extraction."""
    chunk_nodes = await asyncio.gather(
        *[extract_content_nodes(chunk.text, n_rounds) for chunk in data_chunks]
    )

    # Update chunks with their potential nodes
    for chunk, nodes in zip(data_chunks, chunk_nodes):
        chunk.potential_nodes = nodes

    return data_chunks
