"""This module contains the search function that is used to search for nodes in the graph."""

import asyncio
from enum import Enum
from typing import Dict, Any, Callable, List
from pydantic import BaseModel, field_validator

from cognee.modules.search.graph import search_cypher
from cognee.modules.search.graph.search_adjacent import search_adjacent
from cognee.modules.search.vector.search_traverse import search_traverse
from cognee.modules.search.graph.search_summary import search_summary
from cognee.modules.search.graph.search_similarity import search_similarity

from cognee.exceptions import UserNotFoundError
from cognee.shared.utils import send_telemetry
from cognee.modules.users.permissions.methods import get_document_ids_for_user
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User


class SearchType(Enum):
    ADJACENT = "ADJACENT"
    TRAVERSE = "TRAVERSE"
    SIMILARITY = "SIMILARITY"
    SUMMARY = "SUMMARY"
    SUMMARY_CLASSIFICATION = "SUMMARY_CLASSIFICATION"
    NODE_CLASSIFICATION = "NODE_CLASSIFICATION"
    DOCUMENT_CLASSIFICATION = ("DOCUMENT_CLASSIFICATION",)
    CYPHER = "CYPHER"

    @staticmethod
    def from_str(name: str):
        try:
            return SearchType[name.upper()]
        except KeyError as error:
            raise ValueError(f"{name} is not a valid SearchType") from error


class SearchParameters(BaseModel):
    search_type: SearchType
    params: Dict[str, Any]

    @field_validator("search_type", mode="before")
    def convert_string_to_enum(cls, value):  # pylint: disable=no-self-argument
        if isinstance(value, str):
            return SearchType.from_str(value)
        return value


async def search(search_type: str, params: Dict[str, Any], user: User = None) -> List:
    if user is None:
        user = await get_default_user()

    if user is None:
        raise UserNotFoundError

    own_document_ids = await get_document_ids_for_user(user.id)
    search_params = SearchParameters(search_type=search_type, params=params)
    search_results = await specific_search([search_params], user)

    from uuid import UUID

    filtered_search_results = []

    for search_result in search_results:
        document_id = search_result["document_id"] if "document_id" in search_result else None
        document_id = UUID(document_id) if isinstance(document_id, str) else document_id

        if document_id is None or document_id in own_document_ids:
            filtered_search_results.append(search_result)

    return filtered_search_results


async def specific_search(query_params: List[SearchParameters], user) -> List:
    search_functions: Dict[SearchType, Callable] = {
        SearchType.ADJACENT: search_adjacent,
        SearchType.SUMMARY: search_summary,
        SearchType.CYPHER: search_cypher,
        SearchType.TRAVERSE: search_traverse,
        SearchType.SIMILARITY: search_similarity,
    }

    search_tasks = []

    send_telemetry("cognee.search EXECUTION STARTED", user.id)

    for search_param in query_params:
        search_func = search_functions.get(search_param.search_type)
        if search_func:
            # Schedule the coroutine for execution and store the task
            task = search_func(**search_param.params)
            search_tasks.append(task)

    # Use asyncio.gather to run all scheduled tasks concurrently
    search_results = await asyncio.gather(*search_tasks)

    send_telemetry("cognee.search EXECUTION COMPLETED", user.id)

    return search_results[0] if len(search_results) == 1 else search_results
