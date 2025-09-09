from typing import List, Tuple
from pydantic import BaseModel

from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.root_dir import get_absolute_path


class PotentialNodesAndRelationshipNames(BaseModel):
    """Response model containing lists of potential node names and relationship names."""

    nodes: List[str]
    relationship_names: List[str]


async def extract_content_nodes_and_relationship_names(
    content: str, existing_nodes: List[str], n_rounds: int = 2
) -> Tuple[List[str], List[str]]:
    """Extracts node names and relationship_names from content through multiple rounds of analysis."""
    all_nodes: List[str] = existing_nodes.copy()
    all_relationship_names: List[str] = []
    existing_node_set = {node.lower() for node in all_nodes}
    existing_relationship_names = set()

    for round_num in range(n_rounds):
        context = {
            "text": content,
            "potential_nodes": existing_nodes,
            "previous_nodes": all_nodes,
            "previous_relationship_names": all_relationship_names,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        base_directory = get_absolute_path("./tasks/graph/cascade_extract/prompts")
        text_input = render_prompt(
            "extract_graph_relationship_names_prompt_input.txt",
            context,
            base_directory=base_directory,
        )
        system_prompt = read_query_prompt(
            "extract_graph_relationship_names_prompt_system.txt", base_directory=base_directory
        )
        response = await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=PotentialNodesAndRelationshipNames,
        )

        for node in response.nodes:
            if node.lower() not in existing_node_set:
                all_nodes.append(node)
                existing_node_set.add(node.lower())

        for relationship_name in response.relationship_names:
            if relationship_name.lower() not in existing_relationship_names:
                all_relationship_names.append(relationship_name)
                existing_relationship_names.add(relationship_name.lower())

    return all_nodes, all_relationship_names
