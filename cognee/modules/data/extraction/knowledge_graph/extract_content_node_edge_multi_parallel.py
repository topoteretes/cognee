import asyncio
import json
from typing import List, Tuple, Set

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.data_models import KnowledgeGraph, NodeList, EdgeList, Node, Edge


def dedupe_and_normalize_nodes(nodes: List[Node]) -> List[Node]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Node] = []

    for node in nodes:
        node.name = node.name.lower()
        node.type = node.type.lower()

        node.name = node.name.lower().replace("_", " ")
        node.type = node.type.lower().replace("_", " ")

        key = (node.name, node.type)
        if key not in seen:
            seen.add(key)
            out.append(node)

    return out


async def extract_content_node_edge_multi_parallel(content: str, node_rounds: int = 1):
    llm_client = get_llm_client()

    ###### NODE EXTRACTION
    node_prompt_path = "node_extraction_prompt.txt"

    node_system = render_prompt(node_prompt_path, {})

    node_tasks = [
        llm_client.acreate_structured_output(content, node_system, NodeList)
        for _ in range(node_rounds)
    ]

    node_results = await asyncio.gather(*node_tasks)

    all_nodes: List[Node] = [node for nl in node_results for node in nl.nodes]
    ###### NODE DEDUPLICATION
    all_nodes = dedupe_and_normalize_nodes(all_nodes)

    all_nodes_merged = {
        "nodes_to_deduplicate": json.dumps([n.model_dump() for n in all_nodes], ensure_ascii=False)
    }

    merge_system_prompt = "merge_nodes_system_prompt.txt"
    merge_user_prompt = "merge_nodes_user_prompt.txt"

    merge_system = render_prompt(filename=merge_system_prompt, context={})
    merge_user = render_prompt(filename=merge_user_prompt, context=all_nodes_merged)

    final_nodes_list = await llm_client.acreate_structured_output(
        text_input=merge_user, system_prompt=merge_system, response_model=NodeList
    )

    ###### EDGE EXTRACTION

    edge_system_prompt = "edge_extraction_system_prompt.txt"
    edge_user_prompt = "edge_extraction_user_prompt.txt"

    edge_system = render_prompt(edge_system_prompt, {})
    nodes_for_edge_extraction = {
        "final_nodes": json.dumps(
            [n.model_dump() for n in final_nodes_list.nodes], ensure_ascii=False
        ),
        "text": content,
    }

    edge_user = render_prompt(edge_user_prompt, context=nodes_for_edge_extraction)

    final_edges_list = await llm_client.acreate_structured_output(
        text_input=edge_user, system_prompt=edge_system, response_model=EdgeList
    )

    return KnowledgeGraph(nodes=final_nodes_list.nodes, edges=final_edges_list.edges)
