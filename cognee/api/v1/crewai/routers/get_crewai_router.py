import os
import asyncio
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.status import WS_1000_NORMAL_CLOSURE, WS_1008_POLICY_VIOLATION

from cognee.api.DTO import InDTO
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.deletion import prune_data, prune_system
from cognee.modules.data.methods import get_authorized_existing_datasets, load_or_create_datasets
from cognee.modules.data.models import Dataset
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.modules.users.permissions.methods import give_permission_on_dataset
from cognee.modules.users.authentication.default.default_jwt_strategy import DefaultJWTStrategy
from cognee.modules.users.authentication.auth0.auth0_jwt_strategy import Auth0JWTStrategy
from cognee.modules.crewai.get_crewai_pipeline_run_id import get_crewai_pipeline_run_id
from cognee.modules.pipelines.models import PipelineRunInfo, PipelineRunCompleted
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.complex_demos.crewai_demo.src.crewai_demo.main import (
    run_github_ingestion,
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
        # Set context based database settings if necessary
        await set_database_global_context_variables("Github", user.id)

        await prune_data(user)
        await prune_system(user)

        try:
            existing_datasets = await get_authorized_existing_datasets(
                user=user, permission_type="write", datasets=["Github"]
            )
        except PermissionDeniedError:
            print("No datasets were found")
            existing_datasets = []

        datasets = await load_or_create_datasets(["Github"], existing_datasets, user)
        github_dataset: Dataset = next(
            (dataset for dataset in datasets if dataset.name == "Github")
        )

        # Give user proper permissions for dataset
        await give_permission_on_dataset(user, github_dataset.id, "read")
        await give_permission_on_dataset(user, github_dataset.id, "write")
        await give_permission_on_dataset(user, github_dataset.id, "delete")
        await give_permission_on_dataset(user, github_dataset.id, "share")

        await run_github_ingestion(user, github_dataset, payload.username1, payload.username2)

        applicants = {
            "applicant_1": payload.username1,
            "applicant_2": payload.username2,
        }

        run_hiring_crew(user, applicants=applicants, number_of_rounds=2)

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
        dataset_name = "Github"
        await add(payload.feedback, node_set=["final_report"], dataset_name=dataset_name, user=user)
        await cognify(datasets=dataset_name, is_stream_info_enabled=True, user=user)

    @router.websocket("/subscribe")
    async def subscribe_to_crewai_info(websocket: WebSocket):
        await websocket.accept()

        access_token = websocket.cookies.get(os.getenv("AUTH_TOKEN_COOKIE_NAME"))

        try:
            secret = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")

            if os.getenv("USE_AUTH0_AUTHORIZATION") == "True":
                strategy = Auth0JWTStrategy(secret, lifetime_seconds=36000)
            else:
                strategy = DefaultJWTStrategy(secret, lifetime_seconds=3600)

            db_engine = get_relational_engine()

            async with db_engine.get_async_session() as session:
                async with get_user_db_context(session) as user_db:
                    async with get_user_manager_context(user_db) as user_manager:
                        user = await get_authenticated_user(
                            cookie=access_token, strategy_cookie=strategy, user_manager=user_manager
                        )
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
