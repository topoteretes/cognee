from uuid import UUID
from typing import Optional, Union, Literal
from pydantic import ConfigDict, Field
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
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"dataset": "main_dataset", "memoryOnly": True}]},
    )

    data_id: Optional[UUID] = Field(
        default=None,
        examples=[""],
        description="UUID of a single data item to remove. "
        "Requires `dataset` or `datasetId` to also be set.",
    )
    dataset: Optional[str] = Field(
        default=None,
        examples=["default_dataset"],
        description="Dataset name to delete (or clear with memoryOnly). "
        "Provide either `dataset` or `datasetId`, not both.",
    )
    dataset_id: Optional[UUID] = Field(
        default=None,
        examples=[""],
        description="Dataset UUID, alternative to `dataset`. "
        "Provide either `dataset` or `datasetId`, not both.",
    )
    everything: bool = Field(
        default=False,
        description="DANGER: when true, permanently deletes ALL datasets and data the user "
        "owns (relational records, graph, vector embeddings, session cache). "
        "Ignores dataId/dataset/datasetId.",
    )
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
        - Set `dataset` or `datasetId` alone to delete an entire dataset.
        - Set `dataset`/`datasetId` + `dataId` to delete a single item.
        - Set `dataset`/`datasetId` + `memoryOnly: true` to clear memory
          (graph + vector), preserving raw files so the dataset can be re-cognified.
        - Set `dataset`/`datasetId` + `dataId` + `memoryOnly: true` to clear memory
          for a single file only.

        ## Request Parameters
        - **dataId** (Optional[UUID]): UUID of a single data item to remove. Requires
          `dataset` or `datasetId` to also be set.
        - **dataset** (Optional[str]): Name of the dataset to delete or clear.
        - **datasetId** (Optional[UUID]): UUID of the dataset, alternative to `dataset`.
        - **everything** (bool): When true, permanently deletes ALL datasets and data the
          user owns (default: false).
        - **memoryOnly** (bool): When true, delete only memory (graph + vector embeddings),
          preserving raw files and data records (default: false).

        Provide either `dataset` or `datasetId`, not both. Field names are shown camelCased
        in the schema; snake_case aliases (`data_id`, `dataset_id`, `memory_only`) are also
        accepted.

        ## Error Codes
        - **422 Unprocessable Entity**: Invalid parameter combination (e.g. both `dataset`
          and `datasetId`, `dataId` without a dataset, or `memoryOnly` without a dataset)
        - **500 Internal Server Error**: Error during deletion
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
                dataset_id=payload.dataset_id,
                everything=payload.everything,
                memory_only=payload.memory_only,
                user=user,
            )
            return result
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Invalid request parameters. Specify dataset or dataset_id, data_id+dataset, or everything=True."
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
