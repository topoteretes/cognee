import asyncio
from uuid import UUID
from pydantic import BaseModel
from typing import List, Optional
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect

from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted, PipelineRunInfo
from cognee.modules.pipelines.queues.pipeline_run_info_queues import (
    get_from_queue,
    initialize_queue,
    remove_queue,
)


class CognifyPayloadDTO(BaseModel):
    datasets: List[str]
    graph_model: Optional[BaseModel] = KnowledgeGraph


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for the cognitive processing of the content."""
        from cognee.api.v1.cognify import cognify as cognee_cognify

        try:
            cognify_run = await cognee_cognify(
                payload.datasets, user, payload.graph_model, run_in_background=True
            )

            return cognify_run.model_dump()
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.websocket("/subscribe/{pipeline_run_id}")
    async def subscribe_to_cognify_info(websocket: WebSocket, pipeline_run_id: str):
        await websocket.accept()

        auth_message = await websocket.receive_json()

        try:
            await get_authenticated_user(auth_message.get("Authorization"))
        except Exception:
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return

        pipeline_run_id = UUID(pipeline_run_id)

        initialize_queue(pipeline_run_id)

        while True:
            pipeline_run_info = get_from_queue(pipeline_run_id)

            if not pipeline_run_info:
                await asyncio.sleep(2)
                continue

            print(pipeline_run_info)
            if not isinstance(pipeline_run_info, PipelineRunInfo):
                continue

            try:
                await websocket.send_json(jsonable_encoder(pipeline_run_info))

                if isinstance(pipeline_run_info, PipelineRunCompleted):
                    remove_queue(pipeline_run_id)
                    await websocket.close(code=WS_1000_NORMAL_CLOSURE)
                    break
            except WebSocketDisconnect:
                remove_queue(pipeline_run_id)
                break

    return router
