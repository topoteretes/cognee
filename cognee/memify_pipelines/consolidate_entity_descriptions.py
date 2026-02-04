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
from cognee.modules.engine.models import EntityType

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
        selected_fields = ["id", "name", "description", "text", "type"]
        for neighbor in neighbors:
            filtered_neighbor = {k: v for k, v in neighbor.items() if k in selected_fields}
            # if filtered_neighbor contains more fields than just id
            if len(filtered_neighbor.keys()) > 1:
                filtered_neighbors.append(filtered_neighbor)
        allowed_props_keys = {
            "id",
            "description",
            "name",
            "type",
            "ontology_valid",
        }
        filtered_props = {k: v for k, v in props.items() if k in allowed_props_keys}

        return {
            "properties": filtered_props,
            "edges": edges,
            "neighbors": filtered_neighbors,
        }

    return await asyncio.gather(
        *(fetch_neighbour_entities(node_id, props) for node_id, props in entity_nodes)
    )


async def consolidate_entity_descriptions(nodes) -> List[DataPoint]:
    def _load_metadata_to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"index_fields": ["name"]}
        if value is None:
            return {"index_fields": ["name"]}
        return value

    system_prompt = render_prompt(prompt_name, {})

    enriched_data = []
    for node in nodes:
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
        for neighbor in node["neighbors"]:
            if neighbor["type"] == "EntityType":
                entity_type_node = neighbor
                graph_engine = await get_graph_engine()
                node_types, _ = await graph_engine.get_filtered_graph_data(
                    [{"type": ["EntityType"], "name": [entity_type_node["name"]]}]
                )
                if not node_types:
                    continue

                entity_type_id, entity_type_props = node_types[0]
                entity_type_props = {
                    **entity_type_props,
                    "id": entity_type_id,
                    "metadata": _load_metadata_to_dict(entity_type_props.get("metadata")),
                }
                entity_type = EntityType(**entity_type_props)
                entity = Entity(
                    id=UUID(props["id"]),
                    name=props["name"],
                    is_a=entity_type,
                    description=result.description,
                    ontology_valid=props.get("ontology_valid", False),
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
