import asyncio
from uuid import UUID
from pydantic import Field
from typing import Dict, List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect, status
from starlette.status import (
    WS_1000_NORMAL_CLOSURE,
    WS_1008_POLICY_VIOLATION,
    WS_1011_INTERNAL_ERROR,
)

from cognee.api.DTO import InDTO
from cognee.modules.pipelines.methods import get_pipeline_run
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user, get_authenticated_websocket_user
from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_dataset
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import graph_schema_to_graph_model
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunInfo,
    PipelineRunErrored,
    PipelineRunProgress,
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
    chunks_per_batch: Optional[int] = Field(
        default=None,
        examples=[None],
        description=(
            "Number of chunks to process per task batch (e.g. 36). Controls processing "
            "parallelism/throughput; leave null for the pipeline default. Higher the value higher the parallelism/throughput"
        ),
    )
    data_per_batch: Optional[int] = Field(
        default=20,
        examples=[20],
        description="Maximum number of data items to process concurrently within a dataset.",
    )


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        response_model=Dict[UUID, PipelineRunInfo],
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
            user,
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
                # HTTP contract: clients poll run status from the returned run
                # info (and /datasets/status); an errored run is a valid
                # response body here, not an exception.
                raise_on_error=False,
            )

            # If any cognify run errored return JSONResponse with proper error status code
            if any(isinstance(v, PipelineRunErrored) for v in cognify_run.values()):
                first_err = next(
                    (v for v in cognify_run.values() if isinstance(v, PipelineRunErrored)), None
                )
                detail = None
                if first_err is not None:
                    # The failing task's error is carried on ``payload`` (set to
                    # ``repr(error)`` by the pipeline runner); PipelineRunErrored
                    # has no ``error`` attribute. Surface it so the client gets an
                    # actionable message instead of the model's repr.
                    payload = first_err.payload if isinstance(first_err.payload, str) else None
                    detail = payload or str(first_err)

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
    async def subscribe_to_cognify_info(
        websocket: WebSocket,
        pipeline_run_id: str,
        user: Optional[User] = Depends(get_authenticated_websocket_user),
    ):
        """
        Stream one cognify run's progress, then its finished graph.

        ## Path Parameters
        - **pipeline_run_id** (UUID): the run to follow, as returned by
          `POST /cognify`. Must belong to a dataset you can read.

        ## Close Codes
        - **1000**: the run completed and its final payload was sent
        - **1008**: not authenticated, not a valid run id, no such run, or no
          read permission on the run's dataset. A retry replays the same
          rejection, so clients should stop.

        ## Notes
        - The run's update queue is consumed, not observed: one subscriber per
          run. Authorization is therefore checked before the queue is touched
          at all, so a rejected caller cannot disturb the real subscriber.
        """
        await websocket.accept()

        if user is None:
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return

        try:
            run_id = UUID(pipeline_run_id)
        except ValueError:
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Invalid pipeline run id")
            return

        pipeline_run = await get_pipeline_run(run_id)

        if pipeline_run is None:
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Pipeline run not found")
            return

        # Before any queue call, deliberately. Subscribing consumes the run's
        # queue and `initialize_queue` replaces it outright, so an unauthorized
        # caller reaching either one could drain or reset another tenant's run
        # and silently starve its rightful subscriber.
        if not await get_authorized_dataset(user, pipeline_run.dataset_id):
            await websocket.close(
                code=WS_1008_POLICY_VIOLATION,
                reason="Not authorized to read this pipeline run",
            )
            return

        initialize_queue(run_id)

        while True:
            pipeline_run_info = get_from_queue(run_id)

            if not pipeline_run_info:
                await asyncio.sleep(2)
                continue

            if not isinstance(pipeline_run_info, PipelineRunInfo):
                continue

            try:
                # Progress ticks are cheap and frequent — send the event's own
                # fields directly instead of recomputing the (expensive) graph
                # snapshot on every one. That recompute is only meaningful for
                # Started/Yield/Completed/Errored, where the graph has actually
                # changed shape.
                if isinstance(pipeline_run_info, PipelineRunProgress):
                    await websocket.send_json(
                        {
                            "pipeline_run_id": str(pipeline_run_info.pipeline_run_id),
                            "status": pipeline_run_info.status,
                            "completed_items": pipeline_run_info.completed_items,
                            "total_items": pipeline_run_info.total_items,
                            "current_stage": pipeline_run_info.current_stage,
                            "stage_index": pipeline_run_info.stage_index,
                            "stage_total": pipeline_run_info.stage_total,
                        }
                    )
                    continue

                await websocket.send_json(
                    {
                        "pipeline_run_id": str(pipeline_run_info.pipeline_run_id),
                        "status": pipeline_run_info.status,
                        "payload": await get_formatted_graph_data(pipeline_run.dataset_id, user),
                    }
                )

                if isinstance(pipeline_run_info, PipelineRunCompleted):
                    remove_queue(run_id)
                    await websocket.close(code=WS_1000_NORMAL_CLOSURE)
                    break
            except WebSocketDisconnect:
                remove_queue(run_id)
                break
            except DatasetNotFoundError:
                # get_formatted_graph_data re-authorizes on every frame, so
                # this is the dataset being deleted, or read access being
                # revoked, mid-run. The update just popped is already lost;
                # closing beats raising into the ASGI server.
                logger.info("Dataset for pipeline run %s is no longer readable", run_id)
                remove_queue(run_id)
                await websocket.close(
                    code=WS_1008_POLICY_VIOLATION,
                    reason="Not authorized to read this pipeline run",
                )
                break
            except Exception:
                logger.exception("Pipeline run subscription failed for run %s", run_id)
                remove_queue(run_id)
                await websocket.close(code=WS_1011_INTERNAL_ERROR)
                break

    return router
