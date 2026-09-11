from datetime import datetime
from typing import Any, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse, InDTO, OutDTO
from cognee.exceptions import CogneeApiError
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import ContextFormat, SearchResult, SearchType
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry


# Note: Datasets sent by name will only map to datasets owned by the request sender
#       To search for datasets not owned by the request sender dataset UUID is needed
class SearchPayloadDTO(InDTO):
    search_type: SearchType = Field(
        default=SearchType.HYBRID_COMPLETION,
        description=(
            "Retrieval strategy. Common values: HYBRID_COMPLETION (default, passages + entities +"
            " LLM answer), GRAPH_COMPLETION (graph context + LLM answer), CODE (deterministic"
            " code graph), RAG_COMPLETION, CHUNKS, SUMMARIES, TEMPORAL, FEELING_LUCKY"
            " (auto-select), AGENTIC_COMPLETION (enables skills/tools/max_iter)."
        ),
    )
    datasets: Optional[list[str]] = Field(
        default=None,
        examples=[["default_dataset"]],
        description=(
            "Dataset names to search. Names only resolve to datasets owned by the caller;"
            " use dataset_ids for datasets shared with you."
        ),
    )
    dataset_ids: Optional[list[UUID]] = Field(
        default=None,
        examples=[None],
        description=(
            "Dataset UUIDs to search (required for datasets shared with you)."
            " When provided, the datasets name list is ignored."
        ),
    )
    query: str = Field(
        ...,
        examples=["What is in the document?"],
        description="The question to answer. Required; there is no default query.",
    )
    system_prompt: Optional[str] = Field(
        default="Answer the question using the provided context. Be as brief as possible."
    )
    node_name: Optional[list[str]] = Field(
        default=None,
        examples=[None],
        description=(
            "Restrict results to nodes in these node_sets"
            " (the node_set values used during add/remember)."
        ),
    )
    top_k: Optional[int] = Field(default=15)
    only_context: bool = Field(default=False)
    context_format: ContextFormat = Field(
        default=ContextFormat.CONTEXT,
        examples=[ContextFormat.CONTEXT.value],
        description=(
            "Shape of an only_context result. 'context' returns the bare retrieval"
            " context; 'prompt' returns the full envelope a completion would have"
            " received — session guidance, conversation history, and the rendered"
            " user and system prompts. The session layer comes from session_id"
            " (the default session when omitted). Ignored unless only_context is true."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        examples=[None],
        description=(
            "Session whose history and guidance feed the completion (or the"
            " only_context prompt preview). Omit to use the default session."
        ),
    )
    verbose: bool = Field(
        default=False,
        description=(
            "Return detailed result information including the graph representation when available."
        ),
    )
    skills: Optional[list[str]] = Field(
        default=None,
        examples=[None],
        description=(
            "Skill names to load into the agentic retriever."
            " Requires search_type=AGENTIC_COMPLETION; leave null otherwise."
        ),
    )
    tools: Optional[list[str]] = Field(
        default=None,
        examples=[None],
        description=(
            "Whitelist of tool names available to the agentic retriever."
            " Requires search_type=AGENTIC_COMPLETION."
        ),
    )
    max_iter: Optional[int] = Field(
        default=None,
        examples=[None],
        description=(
            "Maximum agentic tool-call iterations before forcing a final answer"
            " (positive integer; AGENTIC_COMPLETION only)."
        ),
    )
    include_references: bool = Field(
        default=False,
        description="Attach source references to completion-type results.",
    )
    code_query: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured arguments for search_type=CODE. Set operation to query_facts, "
            "explore, traverse, find_path, impact_analysis, insights, architecture, or "
            "delta. Add diagram='mermaid' (or 'dot', or true) to receive the result "
            "rendered as diagram source under search_result[0].diagram; architecture "
            "includes a Mermaid diagram unless diagram=false."
        ),
    )


def get_search_router() -> APIRouter:
    router = APIRouter()

    class SearchHistoryItem(OutDTO):
        id: UUID
        text: str
        user: str
        created_at: datetime
        # Null when the search was not scoped to a single dataset.
        dataset_id: Optional[UUID] = None

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
            user,
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
        - **search_type** (SearchType): Type of search to perform (default: HYBRID_COMPLETION). Use AGENTIC_COMPLETION to enable skills, tools and max_iter.
        - **datasets** (Optional[List[str]]): List of dataset names to search within
        - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to search within
        - **query** (str): The search query string
        - **system_prompt** Optional[str]: System prompt to be used for Completion type searches in Cognee
        - **node_name** Optional[list[str]]: Filter results to specific node_sets defined in the add pipeline (for targeted search).
        - **top_k** (Optional[int]): Maximum number of results to return (default: 15)
        - **only_context** bool: Set to true to only return context Cognee will be sending to LLM in Completion type searches. This will be returned instead of LLM calls for completion type searches.
        - **context_format** str: Shape of an only_context result — "context" (default, the bare retrieval context) or "prompt" (the full envelope a completion would receive: session guidance, conversation history, and the rendered user and system prompts).
        - **session_id** (Optional[str]): Session whose history and guidance feed the completion or the prompt preview; the default session when omitted.
        - **verbose** (bool): Return detailed result information including the graph representation when available (default: false)
        - **skills** (Optional[List[str]]): Skill names to load into the agentic retriever (AGENTIC_COMPLETION only)
        - **tools** (Optional[List[str]]): Tool whitelist for AGENTIC_COMPLETION searches
        - **max_iter** (Optional[int]): Max agentic iterations, must be >= 1 (AGENTIC_COMPLETION only)
        - **include_references** (bool): Attach source references to completion-type results (default: true)
        - **code_query** (Optional[dict]): Structured operation arguments for CODE search

        ## Response
        Returns a list of search results containing relevant nodes from the graph.

        ## Error Codes
        - **402/403/404/409/422**: Cognee errors (payment required, permission
          denied, missing user, session-dataset conflict, prerequisites not met)
          return their own status code and message via the global error handler
        - **500 Internal Server Error**: Unexpected error during search

        ## Notes
        - Datasets sent by name will only map to datasets owned by the request sender
        - To search datasets not owned by the request sender, dataset UUID is needed
        - If dataset_ids is provided, the datasets name list is ignored
        """
        send_telemetry(
            "Search API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/search",
                "search_type": str(payload.search_type),
                "datasets": payload.datasets,
                "dataset_ids": [str(dataset_id) for dataset_id in payload.dataset_ids or []],
                # Request fields are recorded by size, matching the recall
                # endpoint's convention (see recall.py telemetry).
                "query": len(payload.query or ""),
                "system_prompt": len(payload.system_prompt or ""),
                "node_name": len(payload.node_name or []),
                "top_k": payload.top_k,
                "only_context": payload.only_context,
                "context_format": payload.context_format,
                "session_id": payload.session_id,
                "verbose": payload.verbose,
                "skills": payload.skills,
                "tools": payload.tools,
                "max_iter": payload.max_iter,
                "include_references": payload.include_references,
                "code_query": len(str(payload.code_query)) if payload.code_query else 0,
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
                context_format=payload.context_format,
                session_id=payload.session_id,
                skills=payload.skills,
                tools=payload.tools,
                max_iter=payload.max_iter,
                include_references=payload.include_references,
                code_query=payload.code_query,
            )

            return jsonable_encoder(results)
        except CogneeApiError:
            # Cognee errors (permission denied, payment required, prerequisites,
            # session-dataset conflicts, ...) carry their own status code and
            # actionable message; the global handler in cognee/api/client.py
            # returns them to the caller.
            raise
        except Exception as error:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(error),
                ).model_dump(),
            )

    return router
