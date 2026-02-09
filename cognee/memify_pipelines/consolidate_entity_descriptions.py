import asyncio
import logging
from uuid import UUID

import cognee
from typing import Any, Dict, List, Optional, Set

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


# region get_entities_with_neighborhood helper functions
async def get_all_entity_nodes(graph_engine):
    """Retrieve all nodes of type Entity from the graph."""
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    return nodes


async def get_entity_neighborhood(
    node_id: str, props: Dict[str, Any], graph_engine
) -> Dict[str, Any]:
    """Fetch and format data for a single entity node."""
    edges, neighbors = await asyncio.gather(
        graph_engine.get_edges(node_id),
        graph_engine.get_neighbors(node_id),
    )

    return {
        "properties": get_entity_properties(props),
        "edges": format_edges(edges),
        "neighbors": format_neighbors(neighbors),
    }


def get_entity_properties(
    props: Dict[str, Any], properties: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Keep only relevant entity properties."""
    if properties is None:
        properties = {"id", "description", "name"}
    return {k: v for k, v in props.items() if k in properties}


def format_edges(edges: List[Any]) -> Dict[str, str]:
    """Map target node IDs to relationship names."""
    return {edge[1]: edge[2]["relationship_name"] for edge in edges}


def format_neighbors(
    neighbors: List[Dict[str, Any]], node_fields: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """Filter neighbor fields and exclude those with only an ID."""
    if node_fields is None:
        node_fields = {"id", "name", "description", "text", "type"}

    filtered_neighbors = [
        {k: v for k, v in neighbor.items() if k in node_fields} for neighbor in neighbors
    ]
    return [n for n in filtered_neighbors if len(n) > 1]


# endregion


async def get_entities_with_neighborhood(args) -> List[Dict[str, Any]]:
    """Iterate through all Entity nodes and fetch their edges and neighbor nodes."""
    graph_engine = await get_graph_engine()
    entity_nodes = await get_all_entity_nodes(graph_engine)

    get_entity_neighborhood_tasks = (
        get_entity_neighborhood(node_id, props, graph_engine) for node_id, props in entity_nodes
    )

    return await asyncio.gather(*get_entity_neighborhood_tasks)


# region consolidate_entity_descriptions helper functions
def load_metadata_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"index_fields": ["name"]}
    if value is None:
        return {"index_fields": ["name"]}
    return value


def build_node_neighborhood_prompt(node):
    props = node["properties"]

    text = (
        "This node's description is the following: "
        + props["name"]
        + " - "
        + props["description"]
        + ". It is connected to it's neighbors in the following way:"
    )
    for neighbor in node["neighbors"]:
        edge_label = node.get("edges", {}).get(neighbor.get("id"), "related to")
        neighbor_name = neighbor.get("name", "")
        neighbor_desc = neighbor.get("description", "")
        if neighbor_desc:
            text += f"\n- {edge_label}: {neighbor_name} - {neighbor_desc}"
        else:
            text += f"\n- {edge_label} - {neighbor.get('text', '')}"

    return text


async def query_LLM(text_input, system_prompt):
    return await LLMGateway.acreate_structured_output(
        text_input=text_input,
        system_prompt=system_prompt,  # no format()
        response_model=NodeDescription,
    )


async def get_entity_type(graph_engine, entity_type_node):
    node_types, _ = await graph_engine.get_filtered_graph_data(
        [{"type": ["EntityType"], "name": [entity_type_node["name"]]}]
    )
    return node_types


def build_entity_type(node_types):
    if not node_types:
        return None
    if len(node_types) > 1:
        logging.warning("Multiple is_a relations found, using only the first one.")
    entity_type_id, entity_type_props = node_types[0]
    entity_type_props = {
        **entity_type_props,
        "id": entity_type_id,
        "metadata": load_metadata_to_dict(entity_type_props.get("metadata")),
    }
    entity_type = EntityType(**entity_type_props)
    return entity_type


def build_entity(id, name, entity_type, description):
    return Entity(
        id=UUID(id),
        name=name,
        is_a=entity_type,
        description=description,
    )


# endregion


async def consolidate_entity_descriptions(nodes) -> List[DataPoint]:
    system_prompt = render_prompt(prompt_name, {})
    enriched_data = []
    graph_engine = await get_graph_engine()

    for node in nodes:
        props = node["properties"]

        text = build_node_neighborhood_prompt(node)

        result = await query_LLM(text, system_prompt)

        for neighbor in node["neighbors"]:
            if neighbor["type"] != "EntityType":
                continue

            entity_type_node = neighbor
            node_types = await get_entity_type(graph_engine, entity_type_node)

            if not node_types:
                continue

            entity_type = build_entity_type(node_types)

            entity = build_entity(props["id"], props["name"], entity_type, result.description)

            enriched_data.append(entity)

    return enriched_data


async def consolidate_entity_descriptions_pipeline():
    extraction_tasks = [Task(get_entities_with_neighborhood)]

    enrichment_tasks = [
        Task(consolidate_entity_descriptions),
        Task(add_data_points),
    ]

    await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # A placeholder to prevent fetching the entire graph
    )
