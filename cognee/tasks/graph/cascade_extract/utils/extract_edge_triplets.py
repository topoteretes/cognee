from typing import List, Tuple
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.shared.data_models import KnowledgeGraph
from cognee.root_dir import get_absolute_path


async def extract_edge_triplets(
    content: str, nodes: List[str], relationship_names: List[str], n_rounds: int = 2
) -> KnowledgeGraph:
    """Creates a knowledge graph by identifying relationships between the provided nodes."""
    llm_client = get_llm_client()
    final_graph = KnowledgeGraph(nodes=[], edges=[])
    existing_nodes = set()
    existing_node_ids = set()
    existing_edge_triplets = set()

    for round_num in range(n_rounds):
        context = {
            "text": content,
            "potential_nodes": nodes,
            "potential_relationship_names": relationship_names,
            "previous_nodes": existing_nodes,
            "previous_edge_triplets": existing_edge_triplets,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        base_directory = get_absolute_path("./tasks/graph/cascade_extract/prompts")
        text_input = render_prompt(
            "extract_graph_edge_triplets_prompt_input.txt", context, base_directory=base_directory
        )
        system_prompt = read_query_prompt(
            "extract_graph_edge_triplets_prompt_system.txt", base_directory=base_directory
        )
        extracted_graph = await llm_client.acreate_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=KnowledgeGraph
        )

        for node in extracted_graph.nodes:
            if node.name not in existing_nodes:
                final_graph.nodes.append(node)
                existing_nodes.add(node.name)
                existing_node_ids.add(node.id)

        for edge in extracted_graph.edges:
            edge_key = (edge.source_node_id, edge.target_node_id, edge.relationship_name)
            if edge_key in existing_edge_triplets:
                continue

            if not (
                edge.source_node_id in existing_node_ids
                and edge.target_node_id in existing_node_ids
            ):
                continue

            final_graph.edges.append(edge)
            existing_edge_triplets.add(edge_key)

    return final_graph
