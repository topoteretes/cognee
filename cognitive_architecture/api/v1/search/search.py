""" This module contains the search function that is used to search for nodes in the graph."""
from enum import Enum, auto
from typing import Dict, Any, Callable, List
from cognitive_architecture.modules.search.graph.search_adjacent import search_adjacent
from cognitive_architecture.modules.search.vector.search_similarity import search_similarity
from cognitive_architecture.modules.search.graph.search_categories import search_categories
from cognitive_architecture.modules.search.graph.search_neighbour import search_neighbour


class SearchType(Enum):
    ADJACENT = auto()
    SIMILARITY = auto()
    CATEGORIES = auto()
    NEIGHBOR = auto()


def complex_search(graph, query_params: Dict[SearchType, Dict[str, Any]]) -> List:
    search_functions: Dict[SearchType, Callable] = {
        SearchType.ADJACENT: search_adjacent,
        SearchType.SIMILARITY: search_similarity,
        SearchType.CATEGORIES: search_categories,
        SearchType.NEIGHBOR: search_neighbour,
    }

    results = set()

    for search_type, params in query_params.items():
        search_func = search_functions.get(search_type)
        if search_func:
            search_result = search_func(graph, **params)
            results.update(search_result)

    return list(results)
