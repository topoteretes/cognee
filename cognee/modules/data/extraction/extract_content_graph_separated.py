import json

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.data_models import KnowledgeGraph, NodeList, EdgeList


async def extract_content_graph_separated(content: str, node_rounds: int = 2, edge_rounds=2):
    llm_client = get_llm_client()

    current_nodes = NodeList()

    for pass_idx in range(node_rounds):
        nodes_json = json.dumps([n.model_dump() for n in current_nodes.nodes], ensure_ascii=False)

        node_system = render_prompt("node_extraction_prompt_sequential_system.txt", {})
        node_user = render_prompt(
            "node_extraction_prompt_sequential_user.txt",
            {
                "text": content,
                "nodes": {nodes_json},
                "total_rounds": {node_rounds},
                "round_number": {pass_idx},
            },
        )

        current_nodes = await llm_client.acreate_structured_output(node_user, node_system, NodeList)

    final_nodes = current_nodes
    final_nodes_json = json.dumps([n.model_dump() for n in final_nodes.nodes], ensure_ascii=False)

    current_edges = EdgeList()

    for pass_idx in range(edge_rounds):
        edges_json = json.dumps([n.model_dump() for n in current_edges.edges], ensure_ascii=False)

        edges_system = render_prompt("edge_extraction_prompt_sequential_system.txt", {})
        edges_user = render_prompt(
            "edge_extraction_prompt_sequential_user.txt",
            {
                "text": content,
                "nodes": {final_nodes_json},
                "edges": {edges_json},
                "total_rounds": {node_rounds},
                "round_number": {pass_idx},
            },
        )

        current_edges = await llm_client.acreate_structured_output(
            edges_user, edges_system, EdgeList
        )

    final_edges = current_edges

    return KnowledgeGraph(nodes=final_nodes.nodes, edges=final_edges.edges)
