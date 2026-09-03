from datetime import datetime
from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.recall.recall import RecallResponse
from cognee.api.sse import SSE_MEDIA_TYPE, sse_headers, wants_event_stream
from cognee.api.v1.recall.recall_stream import begin_recall_stream
from cognee.exceptions import CogneeApiError
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import ContextFormat, SearchResult, SearchType
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry


class RecallPayloadDTO(InDTO):
    # Default is HYBRID_COMPLETION. Pass ``search_type: null`` explicitly
    # to opt into auto-routing (the new ``cognee.recall`` default).
    search_type: Optional[SearchType] = Field(
        default=SearchType.HYBRID_COMPLETION,
        description=(
            "Search strategy, e.g. HYBRID_COMPLETION, GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS. "
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
            "Restrict results to these node sets (the node_set values passed to "
            "/v1/add or /v1/remember). Omit to search all nodes."
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
            " user and system prompts. Ignored unless only_context is true."
        ),
    )
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
            "'session_context', 'tools', 'code', 'all', 'auto', or a list of these. "
            "Defaults to 'auto' (session first when session_id is set, else graph). "
            "'tools' and 'code' are explicit opt-in only — never implied by 'auto' or "
            "'all'. 'tools' requires TOOL_CALLS_ENABLED on the server; 'code' runs a "
            "deterministic code-graph query (see code_query) and tags results "
            "_source='code'."
        ),
    )
    tool_connections: Optional[list[str]] = Field(
        default=None,
        examples=[None],
        description=(
            "Names of authorized external database connections for the 'tools' scope. "
            "Omit to use every connection visible to the caller."
        ),
    )
    stream: Optional[bool] = Field(
        default=None,
        description=(
            "Stream the answer as server-sent events. When omitted, the "
            "`Accept` header decides: streaming happens only for a client that "
            "ranks `text/event-stream` above `application/json`, so `*/*` and "
            "the two listed together both stay on the JSON response."
        ),
    )
    tools_trigger: str = Field(
        default="always",
        description=(
            "When the 'tools' scope runs: 'always', or 'on_empty' to query the "
            "external database only when every other requested source returned nothing."
        ),
    )
    code_query: Optional[dict] = Field(
        default=None,
        examples=[None],
        description=(
            "'code' scope only: structured operation and arguments for the "
            "deterministic code-graph query (same format as /v1/search code_query, "
            'e.g. {"operation": "impact_analysis", "seeds": ["UserService"]}). '
            "Omit to run the default 'explore' operation with the query text as seed. "
            "A seed the code graph cannot resolve contributes no results rather than "
            "failing the recall."
        ),
    )
    context_profile: str = Field(
        default="qa",
        description=(
            "Profile to render for the 'session_context' scope: 'qa' (conversational) or "
            "'agent' (tool/workflow). Ignored by other scopes."
        ),
    )
    response_schema: Optional[dict] = Field(
        default=None,
        examples=[None],
        description=(
            "JSON Schema for structured completion output (typically "
            "MyModel.model_json_schema()). The completion is validated against it "
            "and each result carries the validated payload in its 'structured' "
            "field. Supported by completion-style search types only. Structural "
            "subset: objects, primitives, arrays, enums, optionals, $defs "
            "references; value constraints (minLength, ...) are not enforced "
            "server-side."
        ),
    )


def get_recall_router() -> APIRouter:
    router = APIRouter()

    class RecallHistoryItem(OutDTO):
        id: UUID
        text: str
        user: str
        created_at: datetime
        # Null when the recall was not scoped to a single dataset.
        dataset_id: Optional[UUID] = None

    @router.get("", response_model=list[RecallHistoryItem])
    async def get_recall_history(user: User = Depends(get_authenticated_user)):
        """Get search/recall history for the authenticated user."""
        send_telemetry(
            "Recall API Endpoint Invoked",
            user,
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
    async def recall(
        payload: RecallPayloadDTO,
        request: Request,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Recall information from the knowledge graph.

        This is a memory-oriented alias for the search endpoint. All search
        types and options from v1 are supported.

        ## Request Parameters
        Field names are shown camelCased in the schema (e.g. searchType, datasetIds,
        topK); both camelCase and snake_case are accepted.

        - **search_type** (Optional[SearchType]): Type of search to perform
          (default: HYBRID_COMPLETION). Pass null to enable automatic query routing.
        - **datasets** (Optional[List[str]]): Dataset names to search within
        - **dataset_ids** (Optional[List[UUID]]): Dataset UUIDs to search within;
          take precedence over dataset names when both are provided
        - **query** (str): The search query string
        - **system_prompt** (Optional[str]): System prompt for completion searches
        - **node_name** (Optional[List[str]]): Filter to specific node sets
        - **top_k** (Optional[int]): Maximum results (default: 15)
        - **only_context** (bool): Return only the LLM context
        - **context_format** (str): Shape of an only_context result — "context"
          (default, the bare retrieval context) or "prompt" (the full envelope a
          completion would receive: session guidance, conversation history, and the
          rendered user and system prompts)
        - **verbose** (bool): Verbose output
        - **include_references** (bool): Include source/provenance references in
          completion results (default: true)
        - **stream** (Optional[bool]): Stream the answer as server-sent events
          (`text/event-stream`). Defaults to content negotiation on `Accept`.
        - **session_id** (Optional[str]): Session whose cached QA and trace entries
          should be searched
        - **scope** (Optional[str | List[str]]): Memory sources to include: "graph",
          "session", "trace", "session_context", "tools", "code", "all", "auto", or a
          list of these (default: "auto" — session first when session_id is set, else
          graph). "code" is explicit opt-in only and returns deterministic code-graph
          facts tagged _source="code" (e.g. scope=["graph", "code"])
        - **code_query** (Optional[dict]): "code" scope only — operation and arguments
          for the code-graph query (same format as /v1/search code_query); omit for
          the default "explore" with the query text as seed
        - **response_schema** (Optional[dict]): JSON Schema for structured
          completion output; validated results land in each result's
          ``structured`` field. 422 on schemas outside the supported subset.
        - **contextProfile** (str): Profile to render for the 'session_context' scope: 'qa'
          (conversational) or 'agent' (tool/workflow). Ignored by other scopes. Defaults to 'qa'.
        - **toolConnections** (Optional[List[str]]): Names of authorized external database
          connections for the 'tools' scope. Omit to use every connection visible to the caller.
        - **toolsTrigger** (str): When the 'tools' scope runs: 'always', or 'on_empty' to query the
          external database only when every other requested source returned nothing. Defaults to
          'always'.

        ## Error Codes
        - **402/403/404/409/422**: Cognee errors (payment required, permission
          denied, missing user, session-dataset conflict, prerequisites not met) return their own
          status code and message via the global error handler
        - **409 Conflict**: Unexpected non-Cognee error during recall
        """
        send_telemetry(
            "Recall API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/recall",
                "search_type": str(payload.search_type),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.recall import recall as cognee_recall
        from cognee.modules.recall.methods.model_from_json_schema import model_from_json_schema

        response_model = (
            model_from_json_schema(payload.response_schema)
            if payload.response_schema is not None
            else None
        )

        # One call, two transports: the streaming path must not build its own
        # argument list, or the two drift the moment a parameter is added.
        def _run_recall():
            return cognee_recall(
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
                context_format=payload.context_format,
                session_id=payload.session_id,
                scope=payload.scope,
                context_profile=payload.context_profile,
                include_references=payload.include_references,
                response_model=response_model,
                tool_connections=payload.tool_connections,
                tools_trigger=payload.tools_trigger,
                code_query=payload.code_query,
            )

        streaming = wants_event_stream(request.headers.get("accept"), payload.stream)

        try:
            if streaming:
                # Negotiated inside this handler rather than on a route of its
                # own: anything mounted separately would miss the dependencies
                # attached to this path — on Cloud that includes the pre-flight
                # credit guard, so a separate endpoint would answer for free.
                #
                # Inside the same try as the JSON path on purpose. begin_recall_stream
                # waits for the recall to produce output or fail, and re-raises a
                # failure unchanged, so the handlers below give a streamed request
                # the same 402/403/409/422 the JSON one would have received.
                started = await begin_recall_stream(_run_recall)
                return StreamingResponse(
                    started.frames(), media_type=SSE_MEDIA_TYPE, headers=sse_headers()
                )

            results = await _run_recall()
            return jsonable_encoder(results)
        except CogneeApiError:
            # Cognee errors carry their own status code and actionable message;
            # the global handler in cognee/api/client.py returns them.
            raise
        except ValueError as error:
            # normalize_scope rejects unknown scope names with ValueError;
            # surface it as a 422 with the valid values instead of an opaque
            # 409. The message is rebuilt here rather than echoed from the
            # exception: any ValueError raised deeper in the recall path lands
            # in this handler too, and its text must not reach the client.
            from cognee.memory.entries import _VALID_SCOPES

            logger = get_logger()
            logger.warning("Recall request validation failed: %s", error)
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Invalid recall request. If a scope was given, valid values are: "
                    f"{sorted(_VALID_SCOPES)}."
                },
            )
        except Exception as error:
            logger = get_logger()
            logger.error("Recall endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during recall."},
            )

    return router
