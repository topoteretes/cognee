import asyncio
from typing import List, Optional, Type

from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.graph.exceptions.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

logger = get_logger(level=ERROR)


def format_triplets(edges):
    print("\n\n\n")

    def filter_attributes(obj, attributes):
        """Helper function to filter out non-None properties, including nested dicts."""
        result = {}
        for attr in attributes:
            value = getattr(obj, attr, None)
            if value is not None:
                # If the value is a dict, extract relevant keys from it
                if isinstance(value, dict):
                    nested_values = {
                        k: v for k, v in value.items() if k in attributes and v is not None
                    }
                    result[attr] = nested_values
                else:
                    result[attr] = value
        return result

    triplets = []
    for edge in edges:
        node1 = edge.node1
        node2 = edge.node2
        edge_attributes = edge.attributes
        node1_attributes = node1.attributes
        node2_attributes = node2.attributes

        # Filter only non-None properties
        node1_info = {key: value for key, value in node1_attributes.items() if value is not None}
        node2_info = {key: value for key, value in node2_attributes.items() if value is not None}
        edge_info = {key: value for key, value in edge_attributes.items() if value is not None}

        # Create the formatted triplet
        triplet = f"Node1: {node1_info}\nEdge: {edge_info}\nNode2: {node2_info}\n\n\n"
        triplets.append(triplet)

    return "".join(triplets)


async def get_memory_fragment(
    properties_to_project: Optional[List[str]] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
) -> CogneeGraph:
    """Creates and initializes a CogneeGraph memory fragment with optional property projections."""
    graph_engine = await get_graph_engine()
    memory_fragment = CogneeGraph()
    if properties_to_project is None:
        properties_to_project = ["id", "description", "name", "type", "text"]

    try:
        await memory_fragment.project_graph_from_db(
            graph_engine,
            node_properties_to_project=properties_to_project,
            edge_properties_to_project=["relationship_name"],
            node_type=node_type,
            node_name=node_name,
        )
        memory_fragment.dump_metadata_txt(
            file_path="/home/haopn2/cognee-starter/results/memory_fragment.txt", also_print=False
        )
    except EntityNotFoundError:
        pass

    return memory_fragment


async def brute_force_triplet_search(
    query: str,
    user: User = None,
    top_k: int = 5,
    collections: List[str] = None,
    properties_to_project: List[str] = None,
    memory_fragment: Optional[CogneeGraph] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
) -> list:
    if user is None:
        user = await get_default_user()

    retrieved_results = await brute_force_search(
        query,
        user,
        top_k,
        collections=collections,
        properties_to_project=properties_to_project,
        memory_fragment=memory_fragment,
        node_type=node_type,
        node_name=node_name,
    )
    return retrieved_results


async def brute_force_search(
    query: str,
    user: User,
    top_k: int,
    collections: List[str] = None,
    properties_to_project: List[str] = None,
    memory_fragment: Optional[CogneeGraph] = None,
    node_type: Optional[Type] = None,
    node_name: Optional[List[str]] = None,
) -> list:
    """
    Performs a brute force search to retrieve the top triplets from the graph.

    Args:
        query (str): The search query.
        user (User): The user performing the search.
        top_k (int): The number of top results to retrieve.
        collections (Optional[List[str]]): List of collections to query.
        properties_to_project (Optional[List[str]]): List of properties to project.
        memory_fragment (Optional[CogneeGraph]): Existing memory fragment to reuse.

    Returns:
        list: The top triplet results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    
    import time
    start_time = time.time()
    if memory_fragment is None:
        memory_fragment = await get_memory_fragment(
            properties_to_project, node_type=node_type, node_name=node_name
        )
    print("\n Initialized memory fragment in %.2f seconds" % (time.time() - start_time))

    if collections is None:
        collections = [
            "Entity_name",
            "TextSummary_text",
            "EntityType_name",
            "DocumentChunk_text",
        ]

    try:
        start_time = time.time()
        vector_engine = get_vector_engine()
        print(f"Vector engine initialized in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error("Failed to initialize vector engine: %s", e)
        raise RuntimeError("Initialization error") from e

    send_telemetry("cognee.brute_force_triplet_search EXECUTION STARTED", user.id)

    async def search_in_collection(collection_name: str):
        try:
            return await vector_engine.search(
                collection_name=collection_name, query_text=query, limit=10
            )
        except CollectionNotFoundError:
            return []

    try:
        time_start = time.time()
        results = await asyncio.gather(
            *[search_in_collection(collection_name) for collection_name in collections]
        )
        print(f"Vector search took {time.time() - time_start:.2f} seconds")

        if all(not item for item in results):
            return []

        node_distances = {collection: result for collection, result in zip(collections, results)}
        start_time = time.time()
        await memory_fragment.map_vector_distances_to_graph_nodes(node_distances=node_distances)
        print(f"Mapped vector distances to graph nodes in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        await memory_fragment.map_vector_distances_to_graph_edges(vector_engine, query)
        print(f"Mapped vector distances to graph edges in {time.time() - start_time:.2f} seconds")

        # memory_fragment.dump_metadata_txt(file_path="/mnt/disk1/hao_workspace/cognee/memory_fragment.txt", also_print=False)
        start_time = time.time()
        results = await memory_fragment.calculate_top_triplet_importances(k=top_k)
        print(f"Calculated top triplet importances in {time.time() - start_time:.2f} seconds")
        with open("/home/haopn2/cognee-starter/results/cognee_brute_force_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n🔍 Searching for: '{query}'\n")
            f.write(f"Found {len(results)} results:\n")
            for result in results:
                f.write(f"  - {str(result)}\n")
        
        for edge in results:
            node1 = edge.node1 # source node
            node2 = edge.node2 # target node
            edge_attributes = edge.attributes # relationship attributes between node1 and node2
            limit = 5
            # memory_fragment.write_related_chunks_to_file(entity_id=node1.id, file_path="/mnt/disk1/hao_workspace/cognee/related_chunks.txt")

            edges_e = memory_fragment.get_edges_from_node(node2.id)
            related_chunks = []

            for e in edges_e:
                if e.attributes.get("relationship_type"):
                    other = e.node1 if e.node2.id == node2.id else e.node2
                    if other.attributes.get("type") == "DocumentChunk":
                        related_chunks.append(other)

            with open("/home/haopn2/cognee-starter/results/related_chunks.txt", "a", encoding="utf-8") as f:
                f.write("=========================================================================")
                for chunk in related_chunks[:limit]:
                    f.write(f"Chunk ID: {chunk.id}, Name: {chunk.attributes.get('name')},\n")
                    f.write(f"Text: {chunk.attributes.get('text')}\n")
                    f.write(f"Type: {chunk.attributes.get('type')}\n\n")
                    f.write("-" * 40 + "\n")

        send_telemetry("cognee.brute_force_triplet_search EXECUTION COMPLETED", user.id)

        return results

    except CollectionNotFoundError:
        return []
    except Exception as error:
        logger.error(
            "Error during brute force search for user: %s, query: %s. Error: %s",
            user.id,
            query,
            error,
        )
        send_telemetry(
            "cognee.brute_force_triplet_search EXECUTION FAILED", user.id, {"error": str(error)}
        )
        raise error
