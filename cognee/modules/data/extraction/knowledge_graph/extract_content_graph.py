import os
import asyncio
import json
from fileinput import filename
from typing import Type, List, Tuple, Dict, Any, Set

from langchain_experimental.graph_transformers.llm import system_prompt
from pydantic import BaseModel
from streamlit import context

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.llm.config import get_llm_config
from cognee.shared.data_models import KnowledgeGraph, NodeList, EdgeList, Node, Edge


async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()
    llm_config = get_llm_config()

    prompt_path = llm_config.graph_prompt_path

    # Check if the prompt path is an absolute path or just a filename
    if os.path.isabs(prompt_path):
        # directory containing the file
        base_directory = os.path.dirname(prompt_path)
        # just the filename itself
        prompt_path = os.path.basename(prompt_path)
    else:
        base_directory = None

    system_prompt_graph = render_prompt(prompt_path, {}, base_directory=base_directory)

    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt_graph, response_model
    )

    return content_graph


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


def dedupe_and_normalize_edges(edges: List[Edge]) -> List[Edge]:
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Edge] = []

    for edge in edges:
        edge.relationship_name = edge.relationship_name.lower()

        key = (edge.source_node_id, edge.relationship_name, edge.target_node_id)
        if key not in seen:
            seen.add(key)
            out.append(edge)

    return out


async def extract_content_graph2(
    content: str, response_model: Type[BaseModel], node_rounds: int = 1, edge_rounds: int = 1
):
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

    edge_tasks = [
        llm_client.acreate_structured_output(
            text_input=edge_user, system_prompt=edge_system, response_model=EdgeList
        )
        for _ in range(edge_rounds)
    ]

    edge_results = await asyncio.gather(*edge_tasks)

    all_edges: List[Edge] = [edge for nl in edge_results for edge in nl.edges]
    ###### EDGE DEDUPLICATION
    all_edges = dedupe_and_normalize_edges(all_edges)

    all_edges_merged = {
        "edges_to_deduplicate": json.dumps([n.model_dump() for n in all_edges], ensure_ascii=False)
    }

    merge_system_prompt = "merge_edges_system_prompt.txt"
    merge_user_prompt = "merge_edges_user_prompt.txt"

    merge_system = render_prompt(filename=merge_system_prompt, context={})
    merge_user = render_prompt(filename=merge_user_prompt, context=all_edges_merged)

    final_edges_list = await llm_client.acreate_structured_output(
        text_input=merge_user, system_prompt=merge_system, response_model=EdgeList
    )

    return KnowledgeGraph(nodes=final_nodes_list.nodes, edges=final_edges_list.edges)
