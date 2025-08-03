from typing import Any

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
import json

logger = get_logger("triplet_embedding_poc")


def extract_node_data(node_dict):
    """Extract relevant data from a node dictionary."""
    result = {"id": node_dict["id"]}
    if "metadata" not in node_dict:
        return result
    metadata = json.loads(node_dict["metadata"])
    if "index_fields" not in metadata or not metadata["index_fields"]:
        return result
    index_field_name = metadata["index_fields"][0]  # Always one entry
    if index_field_name not in node_dict:
        return result
    result["content"] = node_dict[index_field_name]
    result["index_field_name"] = index_field_name
    return result


async def get_triplets_from_graph_store(data, triplets_batch_size=10) -> Any:
    graph_engine = await get_graph_engine()
    offset = 0
    while True:
        query = f"""
            MATCH (start_node)-[relationship]->(end_node)
            RETURN start_node, relationship, end_node
            SKIP {offset} LIMIT {triplets_batch_size}
            """
        results = await graph_engine.query(query=query)
        if not results:
            break
        payload = [
            {
                "start_node": extract_node_data(result["start_node"]),
                "relationship": result["relationship"][1],
                "end_node": extract_node_data(result["end_node"]),
            }
            for result in results
        ]
        yield payload
        offset += triplets_batch_size


async def add_triplets_to_collection(data) -> None:
    print(data)


async def get_triplet_embedding_tasks() -> list[Task]:
    triplet_embedding_tasks = [
        Task(get_triplets_from_graph_store, triplets_batch_size=100),
        Task(add_triplets_to_collection),
    ]

    return triplet_embedding_tasks


async def triplet_embedding_postprocessing():
    tasks = await get_triplet_embedding_tasks()

    async for result in run_tasks_base(tasks, user=await get_default_user(), data=[]):
        pass
