import os
import asyncio
import json
from typing import Type, List, Tuple, Dict, Any, Set

from pydantic import BaseModel
from streamlit import text_input

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


async def extract_content_graph_sequential(
    content: str, response_model: Type[BaseModel], graph_extraction_rounds: int = 3
):
    llm_client = get_llm_client()

    graph_system_prompt_path = "generate_graph_prompt.txt"
    graph_user_prompt_path = "generate_graph_prompt_user.txt"
    graph_system = render_prompt(graph_system_prompt_path, {})
    graph_user = render_prompt(
        graph_user_prompt_path, {"text": content, "graph": "nodes: [] , edges =[] "}
    )

    initial_knowledge_graph = await llm_client.acreate_structured_output(
        text_input=graph_user, system_prompt=graph_system, response_model=response_model
    )

    """
    merge_system = render_prompt(filename=merge_system_prompt, context={})
    merge_user = render_prompt(filename=merge_user_prompt, context=all_nodes_merged)

    final_nodes_list = await llm_client.acreate_structured_output(
        text_input=merge_user, system_prompt=merge_system, response_model=NodeList
    )
    """

    final_graph = initial_knowledge_graph
    return final_graph
