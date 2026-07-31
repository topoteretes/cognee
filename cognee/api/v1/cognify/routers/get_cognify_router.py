import os
import asyncio
from uuid import UUID
from pydantic import Field, StrictInt
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect, status
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION

from cognee.api.DTO import InDTO
from cognee.modules.pipelines.methods import get_pipeline_run
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.authentication.default.default_jwt_strategy import DefaultJWTStrategy
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import graph_schema_to_graph_model
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunInfo,
    PipelineRunErrored,
)
from cognee.modules.pipelines.queues.pipeline_run_info_queues import (
    get_from_queue,
    initialize_queue,
    remove_queue,
)
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.api.DTO import ErrorResponse

logger = get_logger("api.cognify")


class CognifyPayloadDTO(InDTO):
    # Examples double as the Swagger try-it-out prefill, which is SUBMITTED
    # as-is on Execute — keep them behavior-neutral (empty/None) for every
    # field where a value changes processing.
    datasets: Optional[List[str]] = Field(
        default=None,
        examples=[["default_dataset"]],
        description=(
            "Dataset names to process; resolved against datasets owned by the authenticated user."
        ),
    )
    dataset_ids: Optional[List[UUID]] = Field(
        default=None,
        examples=[[]],
        description=(
            "Dataset UUIDs to process (required for datasets shared with you). "
            "Takes precedence over the datasets name list when both are provided."
        ),
    )
    run_in_background: Optional[bool] = Field(
        default=False,
        description=(
            "If true, the request returns immediately with a pipeline_run_id while the "
            "graph builds server-side — track completion via GET /v1/datasets/status or "
            "the /v1/cognify/subscribe WebSocket. If false, the request blocks until the "
            "knowledge graph is fully built, which can take minutes for large datasets."
        ),
    )
    graph_model: Optional[dict] = Field(
        default=None,
        examples=[{}],
        description=(
            "JSON schema describing a custom graph model for entity extraction, including a "
            "top-level 'title' key. When omitted or {}, the default KnowledgeGraph model is "
            "used — a restrictive schema here can produce an empty graph."
        ),
    )
    custom_prompt: Optional[str] = Field(
        default="",
        examples=[""],
        description=(
            "Replaces the default entity-extraction prompt to steer which entities and "
            "relationships get extracted (e.g. 'Extract entities focusing on technical "
            "concepts and their relationships.'). Leave empty for the default prompt."
        ),
    )
    chunk_size: Optional[int] = Field(
        default=None,
        examples=[None],
        description=(
            "Maximum tokens per chunk (e.g. 4096). Leave null for automatic model-based "
            "sizing. Larger chunks give more context per LLM extraction pass; smaller "
            "chunks give finer-grained extraction at higher LLM cost."
        ),
    )
    ontology_key: Optional[List[str]] = Field(
        default=None,
        examples=[[]],
        description=(
            "Keys of previously uploaded ontologies (see /v1/ontologies) to ground "
            "entity extraction. Leave empty to process without an ontology."
        ),
    )
    chunks_per_batch: Optional[StrictInt] = Field(
        default=None,
        gt=0,
        examples=[None],
        description=(
            "Number of chunks to process per task batch (e.g. 36). Controls processing "
            "parallelism/throughput; leave null for the pipeline default. Higher the value higher the parallelism/throughput"
        ),
    )
    data_per_batch: StrictInt = Field(
        default=20,
        gt=0,
        examples=[20],
        description="Maximum number of data items to process concurrently within a dataset.",
    )


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        response_model=dict,
        responses={
            400: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    @log_usage(function_name="POST /v1/cognify", log_type="api_endpoint")
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Transform datasets into structured knowledge graphs through cognitive processing.

        This endpoint is the core of Cognee's intelligence layer, responsible for converting
        raw text, documents, and data added through the add endpoint into semantic knowledge graphs.
        It performs deep analysis to extract entities, relationships, and insights from ingested content.

        ## Processing Pipeline
        1. Document classification and permission validation
        2. Text chunking and semantic segmentation
        3. Entity extraction using LLM-powered analysis
        4. Relationship detection and graph construction
        5. Vector embeddings generation for semantic search
        6. Content summarization and indexing

        ## Request Parameters
        - **datasets** (Optional[List[str]]): List of dataset names to process. Dataset names are resolved to datasets owned by the authenticated user.
        - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow processing of datasets not owned by the user (if permitted).
        - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to False (blocking).
        - **graph_model** (Optional[dict]): JSON schema describing a custom graph model for entity extraction. When omitted or {}, the default KnowledgeGraph model is used.
        - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If provided, this prompt will be used instead of the default prompts for knowledge graph extraction.
        - **chunk_size** (Optional[int]): Maximum tokens per chunk. If omitted, Cognee chooses
          a size from the configured LLM and embedding limits.
        - **ontology_key** (Optional[List[str]]): Reference to one or more previously uploaded ontology files to use for knowledge graph construction.
        - **chunks_per_batch** (Optional[int]): Number of chunks to process per task batch in Cognify. Uses the pipeline default when omitted.
        - **data_per_batch** (Optional[int]): Maximum number of data items to process concurrently within a dataset. Defaults to 20.

        ## Response
        - **Blocking execution**: Complete pipeline run information with entity counts, processing duration, and success/failure status
        - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring via WebSocket subscription

        ## Error Codes
        - **400 Bad Request**: When neither datasets nor dataset_ids are provided
        - **409 Conflict**: When a referenced ontology_key does not exist
        - **500 Internal Server Error**: When the pipeline run errors (e.g. missing LLM API key, database connection failure, or a dataset that does not exist)

        ## Example Request
        ```json
        {
            "datasets": ["research_papers", "documentation"],
            "run_in_background": false,
            "custom_prompt": "Extract entities focusing on technical concepts and their relationships. Identify key technologies, methodologies, and their interconnections.",
            "ontology_key": ["medical_ontology_v1"]
        }
        ```

        ## Notes
        To cognify data in datasets not owned by the user and for which the current user has write permission,
        the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

        ## Next Steps
        After successful processing, use the search endpoints to query the generated knowledge graph for insights, relationships, and semantic search.
        """
        send_telemetry(
            "Cognify API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/cognify",
                "cognee_version": cognee_version,
            },
        )

        if not payload.datasets and not payload.dataset_ids:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error="No datasets or dataset_ids provided",
                ).model_dump(),
            )

        from cognee.api.v1.cognify import cognify as cognee_cognify
        from cognee.api.v1.ontologies.ontologies import OntologyService

        try:
            datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets
            config_to_use = None

            if payload.ontology_key:
                ontology_service = OntologyService()
                ontology_contents = ontology_service.get_ontology_contents(
                    payload.ontology_key, user
                )

                from cognee.modules.ontology.ontology_config import Config
                from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import (
                    RDFLibOntologyResolver,
                )
                from io import StringIO

                ontology_streams = [StringIO(content) for content in ontology_contents]
                config_to_use: Config = {
                    "ontology_config": {
                        "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_streams)
                    }
                }

            graph_model_schema = payload.graph_model
            custom_prompt = payload.custom_prompt

            if not graph_model_schema:
                graph_model = KnowledgeGraph
            else:
                graph_model = graph_schema_to_graph_model(graph_model_schema)

            cognify_run = await cognee_cognify(
                datasets,
                user,
                graph_model=graph_model,
                config=config_to_use,
                run_in_background=payload.run_in_background,
                custom_prompt=custom_prompt,
                chunk_size=payload.chunk_size,
                chunks_per_batch=payload.chunks_per_batch,
                data_per_batch=payload.data_per_batch,
            )

            # If any cognify run errored return JSONResponse with proper error status code
            if any(isinstance(v, PipelineRunErrored) for v in cognify_run.values()):
                first_err = next(
                    (v for v in cognify_run.values() if isinstance(v, PipelineRunErrored)), None
                )
                detail = None
                if first_err is not None:
                    detail = getattr(first_err, "error", None) or str(first_err)

                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=ErrorResponse(
                        error="Pipeline run errored",
                        detail=detail,
                    ).model_dump(),
                )
            return cognify_run
        except LLMPaymentRequiredError as error:
            return JSONResponse(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                content=ErrorResponse(
                    error="Token budget exhausted",
                    detail=str(error),
                ).model_dump(),
            )
        except ValueError as e:
            # Ontology key not found (OntologyService raises ValueError)
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=ErrorResponse(
                    error=str(e),
                ).model_dump(),
            )

        except Exception as error:
            logger.exception("Cognify failed")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(error),
                ).model_dump(),
            )

    @router.websocket("/subscribe/{pipeline_run_id}")
    async def subscribe_to_cognify_info(websocket: WebSocket, pipeline_run_id: str):
        await websocket.accept()

        access_token = websocket.cookies.get(os.getenv("AUTH_TOKEN_COOKIE_NAME", "auth_token"))

        try:
            secret = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")

            strategy = DefaultJWTStrategy(secret, lifetime_seconds=3600)

            db_engine = get_relational_engine()

            async with db_engine.get_async_session() as session:
                async with get_user_db_context(session) as user_db:
                    async with get_user_manager_context(user_db) as user_manager:
                        user = await get_authenticated_user(
                            cookie=access_token,
                            strategy_cookie=strategy,
                            user_manager=user_manager,
                            bearer=None,
                        )
        except Exception as error:
            logger.error(f"Authentication failed: {str(error)}")
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return

        pipeline_run_id = UUID(pipeline_run_id)

        pipeline_run = await get_pipeline_run(pipeline_run_id)

        initialize_queue(pipeline_run_id)

        while True:
            pipeline_run_info = get_from_queue(pipeline_run_id)

            if not pipeline_run_info:
                await asyncio.sleep(2)
                continue

            if not isinstance(pipeline_run_info, PipelineRunInfo):
                continue

            try:
                await websocket.send_json(
                    {
                        "pipeline_run_id": str(pipeline_run_info.pipeline_run_id),
                        "status": pipeline_run_info.status,
                        "payload": await get_formatted_graph_data(pipeline_run.dataset_id, user),
                    }
                )

                if isinstance(pipeline_run_info, PipelineRunCompleted):
                    remove_queue(pipeline_run_id)
                    await websocket.close(code=WS_1000_NORMAL_CLOSURE)
                    break
            except WebSocketDisconnect:
                remove_queue(pipeline_run_id)
                break

    return router
