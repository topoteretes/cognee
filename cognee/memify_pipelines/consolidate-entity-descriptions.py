import asyncio
from uuid import UUID

import cognee
from cognee.api.v1.search import SearchType
from typing import Any, Dict, List

import json
from os import path
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.engine.models.Entity import Entity
from cognee.tasks.storage import add_data_points
from cognee.api.v1.visualize.visualize import visualize_graph

custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with the relationship "lives_in".
Ignore all other entities.
"""


async def fetch_entity_neighbors(args) -> List[Dict[str, Any]]:
    """Iterate through all Entity nodes and fetch their edges and neighbor nodes."""
    graph_engine = await get_graph_engine()
    entity_nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])

    entity_neighbors: List[Dict[str, Any]] = []
    for node_id, props in entity_nodes:
        edges = await graph_engine.get_edges(node_id)
        neighbors = await graph_engine.get_neighbors(node_id)

        entity_neighbors.append(
            {
                "id": node_id,
                "properties": props,
                "edges": edges,
                "neighbors": neighbors,
            }
        )

    return entity_neighbors


class NodeDescription(BaseModel):
    description: str


async def enrich_data(param) -> List[DataPoint]:
    system_prompt = (
        "You are a top-tier summarization engine. Your task is to summarize text and make it versatile."
        "Be brief and concise, but keep the important information and the subject."
        "Use synonym words where possible in order to change the wording but keep the meaning."
        "You are to use description provided in the node, as well as data about its neighbors and edges connecting them"
    )
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
            # is_a=entity_type,
            ontology_valid=props.get("ontology_valid", False),
            metadata={"index_fields": ["name"]},
        )
        enriched_data.append(entity)
    return enriched_data


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(
        [
            "Alice moved to Paris in 2010, while Bob has always lived in New York.",
            "Andreas was born in Venice, but later settled in Lisbon.",
            "Diana and Tom were born and raised in Helsinki. Diana currently resides in Berlin, while Tom never moved.",
        ]
    )
    await cognee.cognify(custom_prompt=custom_prompt)

    await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where does Alice live?",
    )

    graph_visualization_path = path.join(path.dirname(__file__), "before_enrichment.html")
    await visualize_graph(graph_visualization_path)

    extraction_tasks = [Task(fetch_entity_neighbors)]

    enrichment_tasks = [
        Task(enrich_data),
        Task(add_data_points),
    ]

    await cognee.memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=[{}],  # A placeholder to prevent fetching the entire graph
    )

    graph_visualization_path = path.join(path.dirname(__file__), "after_enrichment.html")
    await visualize_graph(graph_visualization_path)


if __name__ == "__main__":
    asyncio.run(main())
