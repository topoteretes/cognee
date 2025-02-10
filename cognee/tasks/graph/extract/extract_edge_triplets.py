from typing import List, Tuple
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt, read_query_prompt
from cognee.shared.data_models import KnowledgeGraph


async def extract_edge_triplets(
    content: str, nodes: List[str], relationship_names: List[str], n_rounds: int = 2
) -> KnowledgeGraph:
    """Creates a knowledge graph by identifying relationships between the provided nodes."""
    llm_client = get_llm_client()
    final_graph = KnowledgeGraph(nodes=[], edges=[])
    existing_nodes = set()
    existing_edge_triplets = set()

    for round_num in range(n_rounds):
        context = {
            "text": content,
            "potential_nodes": nodes,
            "potential_relationship_names": relationship_names,
            "previous_nodes": [node.name for node in final_graph.nodes],
            "previous_edge_triplets": [
                (edge.source_node_id, edge.target_node_id, edge.relationship_name)
                for edge in final_graph.edges
            ],
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        text_input = render_prompt("extract_graph_edge_triplets_prompt_input.txt", context)
        system_prompt = read_query_prompt("extract_graph_edge_triplets_prompt_system.txt")
        round_graph = await llm_client.acreate_structured_output(
            text_input=text_input, system_prompt=system_prompt, response_model=KnowledgeGraph
        )

        for node in round_graph.nodes:
            if node.name not in existing_nodes:
                final_graph.nodes.append(node)
                existing_nodes.add(node.name)

        for edge in round_graph.edges:
            edge_key = (edge.source_node_id, edge.target_node_id, edge.relationship_name)
            if edge_key in existing_edge_triplets:
                continue

            source_node_exists = any(node.id == edge.source_node_id for node in final_graph.nodes)
            target_node_exists = any(node.id == edge.target_node_id for node in final_graph.nodes)
            if not (source_node_exists and target_node_exists):
                continue

            final_graph.edges.append(edge)
            existing_edge_triplets.add(edge_key)

    return final_graph
