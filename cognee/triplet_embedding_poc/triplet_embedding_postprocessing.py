from typing import Any, List
import uuid
import json
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.infrastructure.engine import DataPoint


logger = get_logger("triplet_embedding_poc")


def create_triplet_data_point(triplet: dict) -> "TripletDataPoint":
    start_node = triplet.get("start_node", None)
    if start_node:
        start_node_string = start_node.get("content", None)
    else:
        start_node_string = ""

    relationship = triplet.get("relationship", "")

    end_node = triplet.get("end_node", None)
    if end_node:
        end_node_string = end_node.get("content", None)
    else:
        end_node_string = ""

    start_node_type = triplet.get("start_node_type", "")
    end_node_type = triplet.get("end_node_type", "")

    triplet_str = (
        start_node_string
        + "-"
        + start_node_type
        + "-"
        + relationship
        + "-"
        + end_node_string
        + "-"
        + end_node_type
    )

    triplet_uuid = uuid.uuid5(uuid.NAMESPACE_OID, name=triplet_str)

    return TripletDataPoint(id=triplet_uuid, payload=json.dumps(triplet), text=triplet_str)


class TripletDataPoint(DataPoint):
    """DataPoint for storing graph triplets with embedded text representation."""

    payload: str
    text: str
    metadata: dict = {"index_fields": ["text"]}


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
    counter = 0
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
                "start_node_type": result["start_node"]["type"],
                "relationship": result["relationship"][1],
                "end_node_type": result["end_node"]["type"],
                "end_node": extract_node_data(result["end_node"]),
            }
            for result in results
        ]

        counter += len(payload)
        logger.info("Processed %d triplets", counter)
        yield payload
        offset += triplets_batch_size


async def add_triplets_to_collection(
    triplets_batch: List[dict], collection_name: str = "Triplets"
) -> None:
    vector_adapter = get_vector_engine()

    for triplet_batch in triplets_batch:
        data_points = []
        for triplet in triplet_batch:
            try:
                data_point = create_triplet_data_point(triplet)
                data_points.append(data_point)
            except Exception as e:
                raise ValueError(f"Malformed triplet: {triplet}. Error: {e}")

        await vector_adapter.create_data_points(collection_name, data_points)


async def get_triplet_embedding_tasks() -> list[Task]:
    triplet_embedding_tasks = [
        Task(get_triplets_from_graph_store, triplets_batch_size=10),
        Task(add_triplets_to_collection),
    ]

    return triplet_embedding_tasks


async def triplet_embedding_postprocessing():
    tasks = await get_triplet_embedding_tasks()

    async for result in run_tasks_base(tasks, user=await get_default_user(), data=[]):
        pass
