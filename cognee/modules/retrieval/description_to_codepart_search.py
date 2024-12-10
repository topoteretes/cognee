import asyncio
import logging

from typing import Set, List
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry


async def code_description_to_code_part_search(query: str, user: User = None, top_k = 2) -> list:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise PermissionError("No user found in the system. Please create a user.")

    retrieved_codeparts = await code_description_to_code_part(query, user, top_k)
    return retrieved_codeparts



async def code_description_to_code_part(
    query: str,
    user: User,
    top_k: int
) -> List[str]:
    """
    Maps a code description query to relevant code parts using a CodeGraph pipeline.

    Args:
        query (str): The search query describing the code parts.
        user (User): The user performing the search.
        top_k (int): Number of codegraph descriptions to match ( num of corresponding codeparts will be higher)

    Returns:
        Set[str]: A set of unique code parts matching the query.

    Raises:
        ValueError: If arguments are invalid.
        RuntimeError: If an unexpected error occurs during execution.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0 or not isinstance(top_k, int):
        raise ValueError("top_k must be a positive integer.")

    try:
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as init_error:
        logging.error("Failed to initialize engines: %s", init_error, exc_info=True)
        raise RuntimeError("System initialization error. Please try again later.") from init_error

    send_telemetry("code_description_to_code_part_search EXECUTION STARTED", user.id)
    logging.info("Search initiated by user %s with query: '%s' and top_k: %d", user.id, query, top_k)

    try:
        results = await vector_engine.search(
            "code_summary_text", query_text=query, limit=top_k
        )
        if not results:
            logging.warning("No results found for query: '%s' by user: %s", query, user.id)
            return []

        memory_fragment = CogneeGraph()
        await memory_fragment.project_graph_from_db(
            graph_engine,
            node_properties_to_project=['id', 'type', 'text', 'source_code'],
            edge_properties_to_project=['relationship_name']
        )

        code_pieces_to_return = set()

        for node in results:
            node_id = str(node.id)
            node_to_search_from = memory_fragment.get_node(node_id)

            if not node_to_search_from:
                logging.debug("Node %s not found in memory fragment graph", node_id)
                continue

            for code_file in node_to_search_from.get_skeleton_neighbours():
                for code_file_edge in code_file.get_skeleton_edges():
                    if code_file_edge.get_attribute('relationship_name') == 'contains':
                        code_pieces_to_return.add(code_file_edge.get_destination_node())

        logging.info("Search completed for user: %s, query: '%s'. Found %d code pieces.",
                     user.id, query, len(code_pieces_to_return))

        return list(code_pieces_to_return)

    except Exception as exec_error:
        logging.error(
            "Error during code description to code part search for user: %s, query: '%s'. Error: %s",
            user.id, query, exec_error, exc_info=True
        )
        send_telemetry("code_description_to_code_part_search EXECUTION FAILED", user.id)
        raise RuntimeError("An error occurred while processing your request.") from exec_error


if __name__ == "__main__":
    async def main():
        query = "I am looking for a class with blue eyes"
        user = None
        try:
            results = await code_description_to_code_part_search(query, user)
            print("Retrieved Code Parts:", results)
        except Exception as e:
            print(f"An error occurred: {e}")

    asyncio.run(main())


