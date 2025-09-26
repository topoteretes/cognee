from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, Depends, Form
from typing import Optional
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from typing import List
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunErrored,
)

logger = get_logger()


def get_update_router() -> APIRouter:
    router = APIRouter()

    @router.patch("", response_model=None)
    async def update(
        data_id: UUID,
        dataset_id: UUID,
        data: List[UploadFile] = File(default=None),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        user: User = Depends(get_authenticated_user),
    ):
        send_telemetry(
            "Update API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "PATCH /v1/update",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "node_set": str(node_set),
            },
        )

        from cognee.api.v1.update import update as cognee_update

        try:
            update_run = await cognee_update(
                data_id=data_id,
                data=data,
                dataset_id=dataset_id,
                user=user,
                node_set=node_set,
            )

            # If any cognify run errored return JSONResponse with proper error status code
            if any(isinstance(v, PipelineRunErrored) for v in update_run.values()):
                return JSONResponse(status_code=420, content=jsonable_encoder(update_run))
            return update_run

        except Exception as error:
            logger.error(f"Error during deletion by data_id: {str(error)}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
