from typing import Type, List
from pydantic import BaseModel, create_model
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.engine.models.Entity import Entity


async def extract_content_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = render_prompt("generate_graph_prompt.txt", {})
    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph


class EntityListResponse(BaseModel):
    """Response model for entity extraction containing a list of entities."""

    entities: List[Entity]


async def extract_content_nodes(content: str, n_rounds: int = 2) -> List[Entity]:
    """Extracts nodes from content through multiple rounds of analysis."""
    llm_client = get_llm_client()
    all_entities: List[Entity] = []
    existing_names = set()  # Track existing entity names in lowercase

    for round_num in range(n_rounds):
        context = {
            "previous_entities": all_entities,
            "round_number": round_num + 1,
            "total_rounds": n_rounds,
        }

        system_prompt = render_prompt("extract_graph_nodes_prompt.txt", context)
        response = await llm_client.acreate_structured_output(
            text_input=content, system_prompt=system_prompt, response_model=EntityListResponse
        )

        # Only add new entities that haven't been seen before
        for entity in response.entities:
            if entity.name.lower() not in existing_names:
                all_entities.append(entity)
                existing_names.add(entity.name.lower())

    return all_entities
