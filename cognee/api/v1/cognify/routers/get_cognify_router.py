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


logger = get_logger("api.cognify")


class CognifyPayloadDTO(InDTO):
    datasets: Optional[List[str]] = None
    dataset_ids: Optional[List[UUID]] = None
    graph_model: Optional[BaseModel] = KnowledgeGraph
    run_in_background: Optional[bool] = False


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for the cognitive processing of the content."""
        if not payload.datasets and not payload.dataset_ids:
            return JSONResponse(
                status_code=400, content={"error": "No datasets or dataset_ids provided"}
            )

        from cognee.api.v1.cognify import cognify as cognee_cognify

        try:
            datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets

            cognify_run = await cognee_cognify(
                datasets, user, payload.graph_model, run_in_background=payload.run_in_background
            )

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
                        "payload": await get_formatted_graph_data(pipeline_run.dataset_id, user.id),
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
