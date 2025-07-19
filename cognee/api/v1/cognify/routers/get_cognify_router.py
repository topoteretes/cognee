import os
import asyncio
from uuid import UUID
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION

from cognee.api.DTO import InDTO
from cognee.modules.pipelines.methods import get_pipeline_run
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.authentication.default.default_jwt_strategy import DefaultJWTStrategy
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunInfo
from cognee.modules.pipelines.queues.pipeline_run_info_queues import (
    get_from_queue,
    initialize_queue,
    remove_queue,
)
from cognee.shared.logging_utils import get_logger
from cognee.exceptions import (
    CogneeValidationError,
    EmptyDatasetError,
    DatasetNotFoundError,
    MissingAPIKeyError,
    NoDataToProcessError,
)


logger = get_logger("api.cognify")


class CognifyPayloadDTO(InDTO):
    datasets: Optional[List[str]] = None
    dataset_ids: Optional[List[UUID]] = None
    run_in_background: Optional[bool] = False


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
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
        - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to process. UUIDs allow processing of datasets not owned by the user (if permitted).
        - **graph_model** (Optional[BaseModel]): Custom Pydantic model defining the knowledge graph schema. Defaults to KnowledgeGraph for general-purpose processing.
        - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to False (blocking).

        ## Response
        - **Blocking execution**: Complete pipeline run information with entity counts, processing duration, and success/failure status
        - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring via WebSocket subscription

        ## Error Codes
        - **400 Bad Request**: Missing required parameters or invalid input
        - **422 Unprocessable Entity**: No data to process or validation errors
        - **404 Not Found**: Specified datasets don't exist
        - **500 Internal Server Error**: System errors, missing API keys, database connection failures

        ## Example Request
        ```json
        {
            "datasets": ["research_papers", "documentation"],
            "run_in_background": false
        }
        ```

        ## Notes
        To cognify data in datasets not owned by the user and for which the current user has write permission,
        the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

        ## Next Steps
        After successful processing, use the search endpoints to query the generated knowledge graph for insights, relationships, and semantic search.
        """
        # Input validation with enhanced exceptions
        if not payload.datasets and not payload.dataset_ids:
            raise CogneeValidationError(
                message="No datasets or dataset_ids provided",
                user_message="You must specify which datasets to process.",
                suggestions=[
                    "Provide dataset names using the 'datasets' parameter",
                    "Provide dataset UUIDs using the 'dataset_ids' parameter",
                    "Use cognee.datasets() to see available datasets",
                ],
                docs_link="https://docs.cognee.ai/api/cognify",
                context={
                    "provided_datasets": payload.datasets,
                    "provided_dataset_ids": payload.dataset_ids,
                },
                operation="cognify",
            )

        # Check for LLM API key early to provide better error messaging
        llm_api_key = os.getenv("LLM_API_KEY")
        if not llm_api_key:
            raise MissingAPIKeyError(service="LLM", env_var="LLM_API_KEY")

        from cognee.api.v1.cognify import cognify as cognee_cognify

        datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets

        logger.info(
            f"Starting cognify process for user {user.id}",
            extra={
                "user_id": user.id,
                "datasets": datasets,
                "run_in_background": payload.run_in_background,
            },
        )

        # The enhanced exception handler will catch and format any errors from cognee_cognify
        cognify_run = await cognee_cognify(
            datasets, user, run_in_background=payload.run_in_background
        )

        logger.info(
            f"Cognify process completed for user {user.id}",
            extra={"user_id": user.id, "datasets": datasets},
        )

        return cognify_run

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

        try:
            # If the pipeline is already completed, send the completion status
            if isinstance(pipeline_run, PipelineRunCompleted):
                graph_data = await get_formatted_graph_data()
                pipeline_run.payload = {
                    "nodes": graph_data.get("nodes", []),
                    "edges": graph_data.get("edges", []),
                }

                await websocket.send_json(pipeline_run.model_dump())
                await websocket.close(code=WS_1000_NORMAL_CLOSURE)
                return

            # Stream pipeline updates
            while True:
                try:
                    pipeline_run_info = await asyncio.wait_for(
                        get_from_queue(pipeline_run_id), timeout=10.0
                    )

                    if pipeline_run_info:
                        await websocket.send_json(pipeline_run_info.model_dump())

                        if isinstance(pipeline_run_info, PipelineRunCompleted):
                            break
                except asyncio.TimeoutError:
                    # Send a heartbeat to keep the connection alive
                    await websocket.send_json({"type": "heartbeat"})
                except Exception as e:
                    logger.error(f"Error in WebSocket communication: {str(e)}")
                    break

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for pipeline {pipeline_run_id}")
        except Exception as error:
            logger.error(f"WebSocket error: {str(error)}")
        finally:
            remove_queue(pipeline_run_id)

    return router
