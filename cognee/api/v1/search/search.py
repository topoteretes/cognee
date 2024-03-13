""" This module contains the search function that is used to search for nodes in the graph."""
from enum import Enum, auto
from typing import Dict, Any, Callable, List

from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.modules.search.graph.search_adjacent import search_adjacent
from cognee.modules.search.vector.search_similarity import search_similarity
from cognee.modules.search.graph.search_categories import search_categories
from cognee.modules.search.graph.search_neighbour import search_neighbour
from cognee.shared.data_models import GraphDBType
import asyncio

class SearchType(Enum):
    ADJACENT = auto()
    SIMILARITY = auto()
    CATEGORIES = auto()
    NEIGHBOR = auto()


async def search(graph, query_params: Dict[SearchType, Dict[str, Any]]) -> List:
    search_functions: Dict[SearchType, Callable] = {
        SearchType.ADJACENT: search_adjacent,
        SearchType.SIMILARITY: search_similarity,
        SearchType.CATEGORIES: search_categories,
        SearchType.NEIGHBOR: search_neighbour,
    }

    results = []

    # Create a list to hold all the coroutine objects
    search_tasks = []

    for search_type, params in query_params.items():
        search_func = search_functions.get(search_type)
        if search_func:
            # Schedule the coroutine for execution and store the task
            full_params = {**params, 'graph': graph}
            task = search_func(**full_params)
            search_tasks.append(task)

    # Use asyncio.gather to run all scheduled tasks concurrently
    search_results = await asyncio.gather(*search_tasks)

    # Update the results set with the results from all tasks
    for search_result in search_results:
        results.append(search_result)

    return results

if __name__ == "__main__":





    query_params = {
        SearchType.SIMILARITY: {'query': 'your search query here'}
    }
    async def main():
        graph_client = get_graph_client(GraphDBType.NETWORKX)

        await graph_client.load_graph_from_file()
        graph = graph_client.graph
        results = await search(graph, query_params)
        print(results)

    asyncio.run(main())
