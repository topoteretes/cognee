import os
import asyncio
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION

from cognee.api.DTO import InDTO
from cognee.complex_demos.crewai_demo.src.crewai_demo.github_ingest_datapoints import (
    cognify_github_data_from_username,
)
from cognee.modules.crewai.get_crewai_pipeline_run_id import get_crewai_pipeline_run_id
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models import PipelineRunInfo, PipelineRunCompleted
from cognee.complex_demos.crewai_demo.src.crewai_demo.main import (
    # run_github_ingestion,
    run_hiring_crew,
)
from cognee.modules.pipelines.queues.pipeline_run_info_queues import (
    get_from_queue,
    initialize_queue,
    remove_queue,
)


class CrewAIRunPayloadDTO(InDTO):
    username1: str
    username2: str


class CrewAIFeedbackPayloadDTO(InDTO):
    feedback: str


def get_crewai_router() -> APIRouter:
    router = APIRouter()

    @router.post("/run", response_model=bool)
    async def run_crewai(
        payload: CrewAIRunPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        # Run CrewAI with the provided usernames
        # run_future = run_github_ingestion(payload.username1, payload.username2)
        token = os.getenv("GITHUB_TOKEN")

        await cognify_github_data_from_username(payload.username1, token)
        await cognify_github_data_from_username(payload.username2, token)

        applicants = {
            "applicant_1": payload.username1,
            "applicant_2": payload.username2,
        }

        run_hiring_crew(applicants=applicants, number_of_rounds=2)

        return True

    @router.post("/feedback", response_model=None)
    async def send_feedback(
        payload: CrewAIFeedbackPayloadDTO,
        user: User = Depends(
            get_authenticated_user,
        ),
    ):
        from cognee import add, cognify
        # from secrets import choice
        # from string import ascii_letters, digits

        # hash6 = "".join(choice(ascii_letters + digits) for _ in range(6))
        dataset_name = "final_reports"
        await add(payload.feedback, node_set=["final_report"], dataset_name=dataset_name)
        await cognify(datasets=dataset_name, is_stream_info_enabled=True)

    @router.websocket("/subscribe")
    async def subscribe_to_crewai_info(websocket: WebSocket):
        await websocket.accept()

        auth_message = await websocket.receive_json()

        try:
            user = await get_authenticated_user(auth_message.get("Authorization"))
        except Exception:
            await websocket.close(code=WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return

        pipeline_run_id = get_crewai_pipeline_run_id(user.id)

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
                        "payload": pipeline_run_info.payload if pipeline_run_info.payload else None,
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
