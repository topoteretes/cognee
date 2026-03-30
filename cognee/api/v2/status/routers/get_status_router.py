from typing import Optional, Union, List
from datetime import datetime
from pydantic import Field
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.shared.logging_utils import get_logger


class StatusPayloadDTO(InDTO):
    datasets: Optional[List[str]] = Field(default=None)
    items: bool = Field(default=False)
    since: Optional[datetime] = Field(default=None)


def get_status_router() -> APIRouter:
    router = APIRouter()

    @router.post("")
    @log_usage(function_name="POST /v2/status", log_type="api_endpoint")
    async def get_status(payload: StatusPayloadDTO, user: User = Depends(get_authenticated_user)):
        """Return processing status for the user's datasets.

        When `items` is false (default), returns aggregate counts per dataset.
        When `items` is true, returns per-item detail including error messages.
        Use `since` to filter to items created at or after a timestamp.
        """
        send_telemetry(
            "Status API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v2/status",
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v2.status import status

        try:
            results = await status(
                datasets=payload.datasets,
                items=payload.items,
                since=payload.since,
                user=user,
            )
            return jsonable_encoder(results)
        except Exception as error:
            logger = get_logger()
            logger.error("Status endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "An error occurred while fetching status."},
            )

    return router
