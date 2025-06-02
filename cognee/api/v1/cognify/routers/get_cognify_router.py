import asyncio
from uuid import UUID
from typing import List, Optional
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION

from cognee.modules.graph.utils import deduplicate_nodes_and_edges, get_graph_from_model
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
    dataset_ids: Optional[List[UUID]] = None
    graph_model: Optional[BaseModel] = KnowledgeGraph


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for the cognitive processing of the content."""
        from cognee.api.v1.cognify import cognify as cognee_cognify

        try:
            # Send dataset UUIDs if they are given, if not send dataset names
            datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets
            cognify_run = await cognee_cognify(
                datasets, user, payload.graph_model, run_in_background=True
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

            if not isinstance(pipeline_run_info, PipelineRunInfo):
                continue

            try:
                await websocket.send_json(
                    {
                        "pipeline_run_id": str(pipeline_run_info.pipeline_run_id),
                        "status": pipeline_run_info.status,
                        "payload": await get_nodes_and_edges(pipeline_run_info.payload)
                        if pipeline_run_info.payload
                        else None,
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


async def get_nodes_and_edges(data_points):
    nodes = []
    edges = []

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    results = await asyncio.gather(
        *[
            get_graph_from_model(
                data_point,
                added_nodes=added_nodes,
                added_edges=added_edges,
                visited_properties=visited_properties,
            )
            for data_point in data_points
        ]
    )

    for result_nodes, result_edges in results:
        nodes.extend(result_nodes)
        edges.extend(result_edges)

    nodes, edges = deduplicate_nodes_and_edges(nodes, edges)

    return {
        "nodes": list(
            map(
                lambda node: {
                    "id": str(node.id),
                    "label": node.name if hasattr(node, "name") else f"{node.type}_{str(node.id)}",
                    "properties": {},
                },
                nodes,
            )
        ),
        "edges": list(
            map(
                lambda edge: {
                    "source": str(edge[0]),
                    "target": str(edge[1]),
                    "label": edge[2],
                },
                edges,
            )
        ),
    }
