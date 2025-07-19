from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cognee.api.DTO import InDTO
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.data.methods import get_history
from cognee.shared.logging_utils import get_logger
from cognee.exceptions import UnsupportedSearchTypeError, InvalidQueryError, NoDataToProcessError

logger = get_logger()


class SearchPayloadDTO(InDTO):
    search_type: SearchType
    datasets: Optional[List[str]] = None
    dataset_ids: Optional[List[UUID]] = None
    query: str
    top_k: Optional[int] = 10


def get_search_router() -> APIRouter:
    router = APIRouter()

    @router.get("/history", response_model=list)
    async def get_search_history(user: User = Depends(get_authenticated_user)):
        """
        Get search history for the authenticated user.

        This endpoint retrieves the search history for the current user,
        showing previous queries and their results.

        ## Response
        Returns a list of historical search queries and their metadata.

        ## Error Codes
        - **500 Internal Server Error**: Database or system error while retrieving history
        """
        # Remove try-catch to let enhanced exception handler deal with it
        history = await get_history(user.id, limit=0)
        return history

    @router.post("", response_model=list)
    async def search(payload: SearchPayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Search for nodes in the graph database.

        This endpoint performs semantic search across the knowledge graph to find
        relevant nodes based on the provided query. It supports different search
        types and can be scoped to specific datasets.

        ## Request Parameters
        - **search_type** (SearchType): Type of search to perform
        - **datasets** (Optional[List[str]]): List of dataset names to search within
        - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to search within
        - **query** (str): The search query string
        - **top_k** (Optional[int]): Maximum number of results to return (default: 10)

        ## Response
        Returns a list of search results containing relevant nodes from the graph.

        ## Error Codes
        - **400 Bad Request**: Invalid query or search parameters
        - **404 Not Found**: No data found to search
        - **422 Unprocessable Entity**: Unsupported search type
        - **403 Forbidden**: User doesn't have permission to search datasets
        - **500 Internal Server Error**: System error during search

        ## Notes
        - Datasets sent by name will only map to datasets owned by the request sender
        - To search datasets not owned by the request sender, dataset UUID is needed
        - Enhanced error messages provide actionable suggestions for fixing issues
        """
        from cognee.api.v1.search import search as cognee_search

        # Input validation with enhanced exceptions
        if not payload.query or not payload.query.strip():
            raise InvalidQueryError(query=payload.query or "", reason="Query cannot be empty")

        if len(payload.query.strip()) < 2:
            raise InvalidQueryError(
                query=payload.query, reason="Query must be at least 2 characters long"
            )

        # Check if search type is supported
        try:
            search_type = payload.search_type
            logger.info(
                f"Search type validated: {search_type.value}",
                extra={
                    "search_type": search_type.value,
                    "user_id": user.id,
                    "query_length": len(payload.query),
                },
            )
        except ValueError:
            raise UnsupportedSearchTypeError(
                search_type=str(payload.search_type), supported_types=[t.value for t in SearchType]
            )

        # Permission denied errors will be caught and handled by the enhanced exception handler
        # Other exceptions will also be properly formatted by the global handler
        results = await cognee_search(
            query_text=payload.query,
            query_type=payload.search_type,
            user=user,
            datasets=payload.datasets,
            dataset_ids=payload.dataset_ids,
            top_k=payload.top_k,
        )

        # If no results found, that's not necessarily an error, just return empty list
        if not results:
            return []

        return results

    return router
