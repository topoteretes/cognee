import asyncio
from asyncio import gather
from uuid import UUID

import cognee
from typing import Any, Dict, List

import json
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.engine.models.Entity import Entity
from cognee.tasks.storage import add_data_points
from cognee.infrastructure.llm.prompts import render_prompt

prompt_name = "consolidate_entity_details.txt"


class NodeDescription(BaseModel):
    description: str


async def fetch_entity_neighbors(args) -> List[Dict[str, Any]]:
    """Iterate through all Entity nodes and fetch their edges and neighbor nodes."""
    graph_engine = await get_graph_engine()
    entity_nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])

    async def fetch_neighbour_entities(node_id, props):
        """Fetches neighboring nodes and edges from node with node_id"""
        edges, neighbors = await asyncio.gather(
            graph_engine.get_edges(node_id),
            graph_engine.get_neighbors(node_id),
        )
        filtered_neighbors = []
        selected_fields = ["id", "name", "description", "text"]
        for neighbor in neighbors:
            filtered_neighbor = {}
            for selected_field in selected_fields:
                if selected_field in neighbor:
                    filtered_neighbor[selected_field] = neighbor[selected_field]
            # if filtered_neighbor contains more fields than just id
            if len(filtered_neighbor.keys()) > 1:
                filtered_neighbors.append(filtered_neighbor)

        return {
            "properties": props,
            "edges": edges,
            "neighbors": filtered_neighbors,
        }

    return await asyncio.gather(
        *(fetch_neighbour_entities(node_id, props) for node_id, props in entity_nodes)
    )


async def consolidate_entity_descriptions(param) -> List[DataPoint]:
    system_prompt = render_prompt(prompt_name, {})

    enriched_data = []
    for node in param:
        props = node["properties"]
        text = json.dumps(
            {
                "current_description": props.get("description"),
                "neighbors": node["neighbors"],
                "edges": node["edges"],
            },
            default=str,
        )
        result = await LLMGateway.acreate_structured_output(
            text_input=text,
            system_prompt=system_prompt,  # no format()
            response_model=NodeDescription,
        )
        entity = Entity(
            id=UUID(props["id"]),
            name=props["name"],
            description=result.description,
            created_at=props.get("created_at"),
            updated_at=props.get("updated_at"),
            version=props.get("version", 1),
            topological_rank=props.get("topological_rank", 0),
            ontology_valid=props.get("ontology_valid", False),
            metadata=(
                json.loads(props["metadata"])
                if isinstance(props.get("metadata"), str)
                else {"index_fields": ["name"]}
            ),
        )
        enriched_data.append(entity)
    return enriched_data


async def consolidate_entity_descriptions_pipeline():
    extraction_tasks = [Task(fetch_entity_neighbors)]

    enrichment_tasks = [
        Task(consolidate_entity_descriptions),
        Task(add_data_points),
    ]

    await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # A placeholder to prevent fetching the entire graph
    )
