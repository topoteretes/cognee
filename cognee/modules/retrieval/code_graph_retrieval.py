from cognee.low_level import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from .brute_force_triplet_search import brute_force_triplet_search


async def code_graph_retrieval(query: str) -> dict[str, str]:
    subclasses = get_all_subclasses(DataPoint)

    vector_index_collections = []

    for subclass in subclasses:
        index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
        for field_name in index_fields:
            vector_index_collections.append(f"{subclass.__name__}_{field_name}")

    found_triplets = await brute_force_triplet_search(
        query,
        top_k=5,
        collections=vector_index_collections or None,
        properties_to_project=["id", "file_path", "source_code"],
    )

    retrieved_files = {}

    for triplet in found_triplets:
        if triplet.node1.attributes["source_code"]:
            retrieved_files[triplet.node1.attributes["file_path"]] = triplet.node1.attributes[
                "source_code"
            ]
        if triplet.node2.attributes["source_code"]:
            retrieved_files[triplet.node2.attributes["file_path"]] = triplet.node2.attributes[
                "source_code"
            ]

    return retrieved_files
