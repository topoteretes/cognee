from uuid import UUID
from typing import Optional, Union
from pydantic import Field
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse

from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.shared.logging_utils import get_logger


class ForgetPayloadDTO(InDTO):
    data_id: Optional[UUID] = Field(default=None)
    dataset: Optional[Union[str, UUID]] = Field(default=None)
    everything: bool = Field(default=False)
    memory_only: bool = Field(
        default=False,
        description="When True with a dataset, delete only memory (graph nodes/edges and vector embeddings) "
        "and reset pipeline status — raw files and data records are preserved. "
        "This allows re-cognifying the dataset from scratch.",
    )


def get_forget_router() -> APIRouter:
    router = APIRouter()

    @router.post("")
    @log_usage(function_name="POST /v1/forget", log_type="api_endpoint")
    async def forget_endpoint(
        payload: ForgetPayloadDTO, user: User = Depends(get_authenticated_user)
    ):
        """Remove data from the knowledge graph.

        - Set `everything: true` to delete all user data.
        - Set `dataset` alone to delete an entire dataset.
        - Set `dataset` + `data_id` to delete a single item.
        - Set `dataset` + `memory_only: true` to clear memory
          (graph + vector), preserving raw files so the dataset can be re-cognified.
        - Set `dataset` + `data_id` + `memory_only: true` to clear memory
          for a single file only.
        """
        send_telemetry(
            "Forget API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/forget",
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.forget import forget

        try:
            result = await forget(
                data_id=payload.data_id,
                dataset=payload.dataset,
                everything=payload.everything,
                memory_only=payload.memory_only,
                user=user,
            )
            return result
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Invalid request parameters. Specify dataset, data_id+dataset, or everything=True."
                },
            )
        except Exception as error:
            logger = get_logger()
            logger.error("Forget endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "An error occurred during deletion."},
            )

    return router
