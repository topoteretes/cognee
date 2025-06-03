import json
from typing import Type

from pydantic import BaseModel

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.shared.data_models import KnowledgeGraph


async def extract_content_graph_sequential(
    content: str, response_model: Type[BaseModel], graph_extraction_rounds: int = 2
):
    llm_client = get_llm_client()

    graph_system_prompt_path = "generate_graph_prompt_sequential.txt"
    graph_user_prompt_path = "generate_graph_prompt_sequential_user.txt"
    graph_system = render_prompt(graph_system_prompt_path, {})

    current_nodes = []
    current_edges = []

    knowledge_graph = KnowledgeGraph(nodes=[], edges=[])

    for round_idx in range(graph_extraction_rounds):
        nodes_json = json.dumps([n.model_dump() for n in current_nodes], ensure_ascii=False)
        edges_json = json.dumps([e.model_dump() for e in current_edges], ensure_ascii=False)

        graph_user = render_prompt(
            graph_user_prompt_path,
            {
                "text": content,
                "graph": f"nodes: {nodes_json}, edges: {edges_json}",
            },
        )

        knowledge_graph = await llm_client.acreate_structured_output(
            text_input=graph_user,
            system_prompt=graph_system,
            response_model=response_model,
        )

        current_nodes = knowledge_graph.nodes
        current_edges = knowledge_graph.edges

    return knowledge_graph
