from typing import List
from pydantic import BaseModel

from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.root_dir import get_absolute_path


class PotentialNodes(BaseModel):
    """Response model containing a list of potential node names."""

    nodes: List[str]


async def extract_nodes(text: str, n_rounds: int = 2) -> List[str]:
    """Extracts node names from content through multiple rounds of analysis."""
    all_nodes: List[str] = []
    existing_nodes = set()

    for round_num in range(n_rounds):
        context = {
            "previous_nodes": all_nodes,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
            "text": text,
        }
        base_directory = get_absolute_path("./tasks/graph/cascade_extract/prompts")
        text_input = render_prompt(
            "extract_graph_nodes_prompt_input.txt", context, base_directory=base_directory
        )
        system_prompt = read_query_prompt(
            "extract_graph_nodes_prompt_system.txt", base_directory=base_directory
        )
        response = await LLMGateway.acreate_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=PotentialNodes
        )

        for node in response.nodes:
            if node.lower() not in existing_nodes:
                all_nodes.append(node)
                existing_nodes.add(node.lower())

    return all_nodes
