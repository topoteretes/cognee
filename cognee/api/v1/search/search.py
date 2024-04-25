""" This module contains the search function that is used to search for nodes in the graph."""
import asyncio
from enum import Enum
from typing import Dict, Any, Callable, List
from pydantic import BaseModel, field_validator
from cognee.modules.search.graph.search_adjacent import search_adjacent
from cognee.modules.search.vector.search_similarity import search_similarity
from cognee.modules.search.graph.search_categories import search_categories
from cognee.modules.search.graph.search_neighbour import search_neighbour
from cognee.modules.search.graph.search_summary import search_summary
from cognee.infrastructure.databases.graph.get_graph_client import get_graph_client
from cognee.infrastructure import infrastructure_config
from cognee.utils import send_telemetry

class SearchType(Enum):
    ADJACENT = 'ADJACENT'
    SIMILARITY = 'SIMILARITY'
    CATEGORIES = 'CATEGORIES'
    NEIGHBOR = 'NEIGHBOR'
    SUMMARY = 'SUMMARY'

    @staticmethod
    def from_str(name: str):
        try:
            return SearchType[name.upper()]
        except KeyError:
            raise ValueError(f"{name} is not a valid SearchType")

class SearchParameters(BaseModel):
    search_type: SearchType
    params: Dict[str, Any]

    @field_validator('search_type', mode='before')
    def convert_string_to_enum(cls, value):
        if isinstance(value, str):
            return SearchType.from_str(value)
        return value


async def search(search_type: str, params: Dict[str, Any]) -> List:
    search_params = SearchParameters(search_type = search_type, params = params)
    return await specific_search([search_params])


async def specific_search(query_params: List[SearchParameters]) -> List:
    graph_client = await get_graph_client(infrastructure_config.get_config()["graph_engine"])
    graph = graph_client.graph

    search_functions: Dict[SearchType, Callable] = {
        SearchType.ADJACENT: search_adjacent,
        SearchType.SIMILARITY: search_similarity,
        SearchType.CATEGORIES: search_categories,
        SearchType.NEIGHBOR: search_neighbour,
        SearchType.SUMMARY: search_summary
    }

    results = []
    search_tasks = []

    for search_param in query_params:
        search_func = search_functions.get(search_param.search_type)
        if search_func:
            # Schedule the coroutine for execution and store the task
            task = search_func(**search_param.params, graph = graph)
            search_tasks.append(task)

    # Use asyncio.gather to run all scheduled tasks concurrently
    search_results = await asyncio.gather(*search_tasks)

    # Update the results set with the results from all tasks
    results.extend(search_results)

    send_telemetry("cognee.search")

    return results



if __name__ == "__main__":
    async def main():
        # Assuming 'graph' is your graph object, obtained from somewhere
        search_type = 'CATEGORIES'
        params = {'query': 'Ministarstvo', 'other_param': {"node_id": "LLM_LAYER_SUMMARY:DOCUMENT:881ecb36-2819-54c3-8147-ed80293084d6"}}

        results = await search(search_type, params)
        print(results)

    # Run the async main function
    asyncio.run(main())
# if __name__ == "__main__":
#     import asyncio

#     query_params = {
#         SearchType.SIMILARITY: {'query': 'your search query here'}
#     }
#     async def main():
#         graph_client = get_graph_client(GraphDBType.NETWORKX)

#         await graph_client.load_graph_from_file()
#         graph = graph_client.graph
#         results = await search(graph, query_params)
#         print(results)

#     asyncio.run(main())
