"""Task to get triplet datapoints from the graph database as an async generator."""

from typing import AsyncGenerator, Dict, Any, Tuple, List
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.shared.logging_utils import get_logger
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import Triplet

logger = get_logger()


async def get_triplet_datapoints(
    triplets_batch_size: int = 100,
) -> AsyncGenerator[List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]], None]:

    graph_engine = await get_graph_engine()

    if not hasattr(graph_engine, "get_triplets_batch"):
        raise NotImplementedError(
            f"Graph adapter {type(graph_engine).__name__} does not support get_triplets_batch method"
        )

    subclasses = get_all_subclasses(DataPoint)
    datapoint_type_index_property = {}

    for subclass in subclasses:
        if "metadata" in subclass.model_fields:
            metadata_field = subclass.model_fields["metadata"]
            default = getattr(metadata_field, "default", None)
            if isinstance(default, dict):
                index_fields = default.get("index_fields", [])
                if index_fields:
                    datapoint_type_index_property[subclass.__name__] = index_fields


    offset = 0
    while True:
        try:
            triplet_datapoints=[]
            triplets_batch = await graph_engine.get_triplets_batch(
                offset=offset, limit=triplets_batch_size
            )
            if not triplets_batch:
                break


            for triplet_datapoint in triplets_batch:
                node_from = triplet_datapoint.get("start_node", None)
                node_to = triplet_datapoint.get("end_node", None)
                edge = triplet_datapoint.get("relationship_properties", None)

                node_from_type = node_from.get("type", None)
                node_to_type = node_to.get("type", None)

                node_from_embeddable = node_from.get(datapoint_type_index_property.get(node_from_type), None)
                node_to_embeddable = node_to.get(datapoint_type_index_property.get(node_to_type), None)
                edge_embeddable = edge.get(datapoint_type_index_property.get('EdgeType'), None)
                print()




            yield triplets_batch

            offset += len(triplets_batch)
            if len(triplets_batch) < triplets_batch_size:
                break

        except Exception as e:
            logger.error(f"Error retrieving triplet batch at offset {offset}: {e}")
            raise

