from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.recall.recall import RecallResponse
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import SearchResult, SearchType
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError, UserNotFoundError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry


class RecallPayloadDTO(InDTO):
    # Default preserved as GRAPH_COMPLETION for backward compatibility
    # with existing HTTP clients. Pass ``search_type: null`` explicitly
    # to opt into auto-routing (the new ``cognee.recall`` default).
    search_type: Optional[SearchType] = Field(default=SearchType.GRAPH_COMPLETION)
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
    session_id: Optional[str] = Field(default=None)
    scope: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "Which memory sources to include: 'graph', 'session', 'trace', "
            "'graph_context', 'all', or a list. Defaults to 'auto' (session "
            "first when session_id is set, else graph)."
        ),
    )


def get_recall_router() -> APIRouter:
    router = APIRouter()

    class RecallHistoryItem(OutDTO):
        id: UUID
        text: str
        user: str
        created_at: datetime

    @router.get("", response_model=list[RecallHistoryItem])
    async def get_recall_history(user: User = Depends(get_authenticated_user)):
        """Get search/recall history for the authenticated user."""
        send_telemetry(
            "Recall API Endpoint Invoked",
            user.id,
            additional_properties={"endpoint": "GET /v1/recall", "cognee_version": cognee_version},
        )

        try:
            history = await get_history(user.id, limit=0)
            return history
        except Exception as error:
            logger = get_logger()
            logger.error("Recall history error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "An error occurred while fetching recall history."},
            )

    @router.post("", response_model=list[RecallResponse])
    @log_usage(function_name="POST /v1/recall", log_type="api_endpoint")
    async def recall(payload: RecallPayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Recall information from the knowledge graph.

        This is a memory-oriented alias for the search endpoint. All search
        types and options from v1 are supported.

        ## Request Parameters
        - **search_type** (SearchType): Type of search to perform
        - **datasets** (Optional[List[str]]): Dataset names to search within
        - **dataset_ids** (Optional[List[UUID]]): Dataset UUIDs to search within
        - **query** (str): The search query string
        - **system_prompt** (Optional[str]): System prompt for completion searches
        - **node_name** (Optional[List[str]]): Filter to specific node sets
        - **top_k** (Optional[int]): Maximum results (default: 10)
        - **only_context** (bool): Return only the LLM context
        - **verbose** (bool): Verbose output

        ## Error Codes
        - **409 Conflict**: Error during recall
        - **403 Forbidden**: Permission denied (returns empty list)
        """
        send_telemetry(
            "Recall API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/recall",
                "search_type": str(payload.search_type),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.recall import recall as cognee_recall

        try:
            results = await cognee_recall(
                query_text=payload.query,
                query_type=payload.search_type,
                user=user,
                datasets=payload.datasets,
                dataset_ids=payload.dataset_ids,
                system_prompt=payload.system_prompt,
                node_name=payload.node_name,
                top_k=payload.top_k,
                verbose=payload.verbose,
                only_context=payload.only_context,
                session_id=payload.session_id,
                scope=payload.scope,
            )
            return jsonable_encoder(results)
        except (DatabaseNotCreatedError, UserNotFoundError, CogneeValidationError) as e:
            logger = get_logger()
            logger.error("Recall prerequisites error: %s", e, exc_info=True)
            status_code = getattr(e, "status_code", 422)
            return JSONResponse(
                status_code=status_code,
                content={
                    "error": "Recall prerequisites not met",
                    "hint": "Run `await cognee.remember(...)` or `await cognee.add(...)` then `await cognee.cognify()` before recalling.",
                },
            )
        except PermissionDeniedError:
            return []
        except Exception as error:
            logger = get_logger()
            logger.error("Recall endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during recall."},
            )

    return router
