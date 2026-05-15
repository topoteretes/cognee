from datetime import datetime
from typing import Any, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse, InDTO, OutDTO
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import SearchResult, SearchType
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError, UserNotFoundError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry


# Note: Datasets sent by name will only map to datasets owned by the request sender
#       To search for datasets not owned by the request sender dataset UUID is needed
class SearchPayloadDTO(InDTO):
    search_type: SearchType = Field(default=SearchType.GRAPH_COMPLETION)
    datasets: Optional[list[str]] = Field(default=None)
    dataset_ids: Optional[list[UUID]] = Field(default=None, examples=[[]])
    query: str = Field(default="What is in the document?")
    system_prompt: Optional[str] = Field(
        default="Answer the question using the provided context. Be as brief as possible."
    )
    node_name: Optional[list[str]] = Field(default=None, example=[])
    top_k: Optional[int] = Field(default=10)
    only_context: bool = Field(default=False)
    verbose: bool = Field(default=False)


def get_search_router() -> APIRouter:
    router = APIRouter()

    class SearchHistoryItem(OutDTO):
        id: UUID
        text: str
        user: str
        created_at: datetime

    @router.get(
        "",
        response_model=List[SearchHistoryItem],
        responses={
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def get_search_history(user: User = Depends(get_authenticated_user)):
        """
        Get search history for the authenticated user.

        This endpoint retrieves the search history for the authenticated user,
        returning a list of previously executed searches with their timestamps.

        ## Response
        Returns a list of search history items containing:
        - **id**: Unique identifier for the search
        - **text**: The search query text
        - **user**: User who performed the search
        - **created_at**: When the search was performed

        ## Error Codes
        - **500 Internal Server Error**: Error retrieving search history
        """
        send_telemetry(
            "Search API Endpoint Invoked",
            user.id,
            additional_properties={"endpoint": "GET /v1/search", "cognee_version": cognee_version},
        )

        try:
            history = await get_history(user.id, limit=0)

            return history
        except Exception as error:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(error),
                ).model_dump(),
            )

    @router.post(
        "",
        response_model=Union[List[SearchResult], List],
        responses={
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    @log_usage(function_name="POST /v1/search", log_type="api_endpoint")
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
        - **system_prompt** Optional[str]: System prompt to be used for Completion type searches in Cognee
        - **node_name** Optional[list[str]]: Filter results to specific node_sets defined in the add pipeline (for targeted search).
        - **top_k** (Optional[int]): Maximum number of results to return (default: 10)
        - **only_context** bool: Set to true to only return context Cognee will be sending to LLM in Completion type searches. This will be returned instead of LLM calls for completion type searches.

        ## Response
        Returns a list of search results containing relevant nodes from the graph.

        ## Error Codes
        - **409 Conflict**: Error during search operation
        - **403 Forbidden**: User doesn't have permission to search datasets (returns empty list)

        ## Notes
        - Datasets sent by name will only map to datasets owned by the request sender
        - To search datasets not owned by the request sender, dataset UUID is needed
        - If permission is denied, returns empty list instead of error
        """
        send_telemetry(
            "Search API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/search",
                "search_type": str(payload.search_type),
                "datasets": payload.datasets,
                "dataset_ids": [str(dataset_id) for dataset_id in payload.dataset_ids or []],
                "query": payload.query,
                "system_prompt": payload.system_prompt,
                "node_name": payload.node_name,
                "top_k": payload.top_k,
                "only_context": payload.only_context,
                "verbose": payload.verbose,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.search import search as cognee_search

        try:
            results = await cognee_search(
                query_text=payload.query,
                query_type=payload.search_type,
                user=user,
                datasets=payload.datasets
                if not payload.dataset_ids
                else None,  # If dataset_ids are provided, ignore datasets by name to avoid confusion and potential mismatches.
                dataset_ids=payload.dataset_ids,
                system_prompt=payload.system_prompt,
                node_name=payload.node_name,
                top_k=payload.top_k,
                verbose=payload.verbose,
                only_context=payload.only_context,
            )

            return jsonable_encoder(results)
        except PermissionDeniedError as e:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=ErrorResponse(
                    error="Permission denied",
                    detail=str(e),
                ).model_dump(),
            )
        except (DatabaseNotCreatedError, UserNotFoundError, CogneeValidationError) as e:
            status_code = getattr(e, "status_code", status.HTTP_422_UNPROCESSABLE_ENTITY)
            return JSONResponse(
                status_code=status_code,
                content=ErrorResponse(
                    error="Search prerequisites not met, hint: Run `await cognee.add(...)` then `await cognee.cognify()` before searching.",
                    detail=str(e),
                    # Previous hint not matching "Error Response" structure defined in cognee.api.DTO, included in error.
                ).model_dump(),
            )

        except Exception as error:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(error),
                ).model_dump(),
            )

    return router
