from typing import List
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine.models.DataPoint import DataPoint


async def assert_nodes_vector_index_not_present(data_points: List[DataPoint]):
    vector_engine = get_vector_engine()

    data_points_by_vector_collection = {}

    for data_point in data_points:
        node_metadata = data_point.metadata or {}
        collection_name = data_point.type + "_" + node_metadata["index_fields"][0]

        if collection_name not in data_points_by_vector_collection:
            data_points_by_vector_collection[collection_name] = []

        data_points_by_vector_collection[collection_name].append(data_point)

    for collection_name, collection_data_points in data_points_by_vector_collection.items():
        query_data_point_ids = set([str(data_point.id) for data_point in collection_data_points])

        vector_items = await vector_engine.retrieve(collection_name, list(query_data_point_ids))

        for vector_item in vector_items:
            assert str(vector_item.id) not in query_data_point_ids, (
                f"{vector_item.payload['text']} is still present in the vector store."
            )
