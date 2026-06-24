from uuid import UUID
from typing import Optional
from pydantic import ConfigDict, Field
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse

from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("validate_router")


class ValidatePayloadDTO(InDTO):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"dataset": "main_dataset"}]},
    )

    dataset: Optional[str] = Field(
        default="main_dataset",
        examples=["main_dataset"],
        description="Dataset name to validate.",
    )
    dataset_id: Optional[UUID] = Field(
        default=None,
        examples=[""],
        description="Dataset UUID, alternative to `dataset`.",
    )


def get_validate_router() -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def validate_endpoint(
        payload: ValidatePayloadDTO, user: User = Depends(get_authenticated_user)
    ):
        """Validate knowledge graph integrity for a dataset.

        Cross-checks graph, vector, and relational stores to surface
        inconsistencies like unembedded nodes, dangling edges, or
        uncognified data items.

        Returns a ValidationReport with status, issues list, and summary.
        """
        from cognee.api.v1.validate import validate

        dataset_ref = payload.dataset_id or payload.dataset or "main_dataset"

        try:
            report = await validate(dataset=dataset_ref, user=user)
            return report.model_dump()
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": str(exc)},
            )
        except Exception as error:
            logger.error("Validate endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Validation failed due to an internal error."},
            )

    return router
