import asyncio
import logging

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
) -> list:
    """
        Performs codegraph description to code part map for CodeGraph pipeline.

        Args:
            query (str): The search query
            user (User): The user performing the search
            top_k (int): The number of top results to retrieve

        Returns:
            list: Corresponding code pieces to the query.
    """
    if not query or not isinstance(query, str):
        raise ValueError("The query must be a non-empty string.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    try:
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()
    except Exception as e:
        logging.error("Failed to initialize engines: %s", e)
        raise RuntimeError("Initialization error") from e

    send_telemetry("code_description_to_code_part_search EXECUTION STARTED", user.id)

    try:

        results = await vector_engine.search("code_summary_text", query_text=query, limit=top_k)

        memory_fragment = CogneeGraph()

        await memory_fragment.project_graph_from_db(graph_engine,
                                              node_properties_to_project=['id',
                                                                          'type',
                                                                          'text',
                                                                          'source_code'],
                                              edge_properties_to_project=['relationship_name'])

        code_pieces_to_return = set()

        for node in results: # :TODO: This must be changed when the structure of codegraph will change and it will. Now this is the initial version that works well with the actual implementation
            node_to_search_from = memory_fragment.get_node(str(node.id))
            for code_file in node_to_search_from.get_skeleton_neighbours():
                for code_file_edge in code_file.get_skeleton_edges():
                    if code_file_edge.get_attribute('relationship_type') == 'contains':
                        code_pieces_to_return.add(code_file_edge.get_node_to())

        return code_pieces_to_return

    except Exception as e:
        logging.error("Error during description to codepart search for user: %s, query: %s. Error: %s", user.id, query, e)
        send_telemetry("code_description_to_code_part_search EXECUTION FAILED", user.id)
        raise RuntimeError("An error occurred during description to codepart search") from e


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


