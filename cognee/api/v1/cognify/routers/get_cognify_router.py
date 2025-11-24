import os
import asyncio
from uuid import UUID
from pydantic import Field
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
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
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger("api.cognify")


class CognifyPayloadDTO(InDTO):
    datasets: Optional[List[str]] = Field(default=None)
    dataset_ids: Optional[List[UUID]] = Field(default=None, examples=[[]])
    run_in_background: Optional[bool] = Field(default=False)
    custom_prompt: Optional[str] = Field(
        default="", description="Custom prompt for entity extraction and graph generation"
    )


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
        - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow processing of datasets not owned by the user (if permitted).
        - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to False (blocking).
        - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If provided, this prompt will be used instead of the default prompts for knowledge graph extraction.

        ## Response
        - **Blocking execution**: Complete pipeline run information with entity counts, processing duration, and success/failure status
        - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring via WebSocket subscription

        ## Error Codes
        - **400 Bad Request**: When neither datasets nor dataset_ids are provided, or when specified datasets don't exist
        - **409 Conflict**: When processing fails due to system errors, missing LLM API keys, database connection failures, or corrupted content

        ## Example Request
        ```json
        {
            "datasets": ["research_papers", "documentation"],
            "run_in_background": false,
            "custom_prompt": "Extract entities focusing on technical concepts and their relationships. Identify key technologies, methodologies, and their interconnections."
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
                status_code=400, content={"error": "No datasets or dataset_ids provided"}
            )

        from cognee.api.v1.cognify import cognify as cognee_cognify

        try:
            datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets

            cognify_run = await cognee_cognify(
                datasets,
                user,
                run_in_background=payload.run_in_background,
                custom_prompt=payload.custom_prompt,
            )

            # If any cognify run errored return JSONResponse with proper error status code
            if any(isinstance(v, PipelineRunErrored) for v in cognify_run.values()):
                return JSONResponse(status_code=420, content=jsonable_encoder(cognify_run))
            return cognify_run
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

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
