from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO, ErrorResponse
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
    search_type: Optional[SearchType] = Field(
        default=SearchType.GRAPH_COMPLETION,
        description=(
            "Search strategy, e.g. GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES. "
            "Pass null to let cognee auto-route the query to the best strategy."
        ),
    )
    datasets: Optional[list[str]] = Field(
        default=None,
        examples=[["default_dataset"]],
        description=(
            "Dataset names to search within. Omit (null) to search all datasets "
            "you have read access to."
        ),
    )
    dataset_ids: Optional[list[UUID]] = Field(
        default=None,
        examples=[None],
        description=(
            "Dataset UUIDs to search within; takes precedence over 'datasets' names "
            "when both are provided. Leave empty to resolve by name."
        ),
    )
    query: str = Field(default="What is in the document?")
    system_prompt: Optional[str] = Field(
        default="Answer the question using the provided context. Be as brief as possible."
    )
    node_name: Optional[list[str]] = Field(
        default=None,
        examples=[None],
        description=(
            "Restrict results to these node sets (the node_set values passed to "
            "/v1/add or /v1/remember). Omit to search all nodes."
        ),
    )
    top_k: Optional[int] = Field(default=15)
    only_context: bool = Field(default=False)
    verbose: bool = Field(default=False)
    include_references: bool = Field(
        default=False,
        description="Include source/provenance references in completion results.",
    )
    session_id: Optional[str] = Field(
        default=None,
        examples=[None],
        description=(
            "Session whose cached QA and trace entries should be searched. With "
            "search_type null and no datasets, session hits short-circuit the "
            "graph search."
        ),
    )
    scope: Optional[Union[str, list[str]]] = Field(
        default=None,
        examples=[None],
        description=(
            "Which memory sources to include: 'graph', 'session', 'trace', "
            "'graph_context', 'session_context', 'all', 'auto', or a list of these. Defaults "
            "to 'auto' (session first when session_id is set, else graph)."
        ),
    )
    context_profile: str = Field(
        default="qa",
        description=(
            "Profile to render for the 'session_context' scope: 'qa' (conversational) or "
            "'agent' (tool/workflow). Ignored by other scopes."
        ),
    )


def get_recall_router() -> APIRouter:
    router = APIRouter()

    class RecallHistoryItem(OutDTO):
        id: UUID
        text: str
        user: str
        created_at: datetime

    @router.get(
        "",
        response_model=list[RecallHistoryItem],
        responses={
            500: {"model": ErrorResponse},
        },
    )
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
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="An error occurred while fetching recall history.",
                ).model_dump(),
            )

    @router.post(
        "",
        response_model=list[RecallResponse],
        responses={
            403: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    @log_usage(function_name="POST /v1/recall", log_type="api_endpoint")
    async def recall(payload: RecallPayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Recall information from the knowledge graph.

        This is a memory-oriented alias for the search endpoint. All search
        types and options from v1 are supported.

        ## Request Parameters
        Field names are shown camelCased in the schema (e.g. searchType, datasetIds,
        topK); both camelCase and snake_case are accepted.

        - **search_type** (Optional[SearchType]): Type of search to perform
          (default: GRAPH_COMPLETION). Pass null to enable automatic query routing.
        - **datasets** (Optional[List[str]]): Dataset names to search within
        - **dataset_ids** (Optional[List[UUID]]): Dataset UUIDs to search within;
          take precedence over dataset names when both are provided
        - **query** (str): The search query string
        - **system_prompt** (Optional[str]): System prompt for completion searches
        - **node_name** (Optional[List[str]]): Filter to specific node sets
        - **top_k** (Optional[int]): Maximum results (default: 15)
        - **only_context** (bool): Return only the LLM context
        - **verbose** (bool): Verbose output
        - **include_references** (bool): Include source/provenance references in
          completion results (default: true)
        - **session_id** (Optional[str]): Session whose cached QA and trace entries
          should be searched
        - **scope** (Optional[str | List[str]]): Memory sources to include: "graph",
          "session", "trace", "graph_context", "all", "auto", or a list of these
          (default: "auto" — session first when session_id is set, else graph)

        ## Error Codes
        - **500 Internal Server Error**: Error during recall
        - **403 Forbidden**: Permission denied (returns empty list)
        - **422 Unprocessable Entity**: Recall prerequisites not met — ingest data
          first (POST /v1/remember or /v1/add followed by /v1/cognify)
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
                context_profile=payload.context_profile,
                include_references=payload.include_references,
            )
            return jsonable_encoder(results)
        except (DatabaseNotCreatedError, UserNotFoundError, CogneeValidationError) as e:
            logger = get_logger()
            logger.error("Recall prerequisites error: %s", e, exc_info=True)
            status_code = getattr(e, "status_code", 422)
            return JSONResponse(
                status_code=status_code,
                content=ErrorResponse(
                    error="Recall prerequisites not met",
                    detail="Run `await cognee.remember(...)` or `await cognee.add(...)` then `await cognee.cognify()` before recalling.",
                ).model_dump(),
            )
        except PermissionDeniedError:
            return []
        except Exception as error:
            logger = get_logger()
            logger.error("Recall endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="An error occurred during recall.",
                    detail=str(error),
                ).model_dump(),
            )

    return router
