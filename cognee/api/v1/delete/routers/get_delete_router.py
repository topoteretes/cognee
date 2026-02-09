from uuid import UUID
from deprecated import deprecated
from fastapi import Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter

from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


def get_delete_router() -> APIRouter:
    router = APIRouter()

    @router.delete("", response_model=None)
    @deprecated(
        reason="DELETE /v1/delete is deprecated. Use DELETE /v1/datasets/{dataset_id}/data/{data_id} instead.",
        version="0.3.9",
    )
    async def delete(
        data_id: UUID,
        dataset_id: UUID,
        mode: str = "soft",
        user: User = Depends(get_authenticated_user),
    ):
        """Delete data by its ID from the specified dataset.

        Args:
            data_id: The UUID of the data to delete
            dataset_id: The UUID of the dataset containing the data
            mode: "soft" (default) or "hard" - hard mode also deletes degree-one entity nodes
            user: Authenticated user

        Returns:
            JSON response indicating success or failure

        """
        send_telemetry(
            "Delete API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "DELETE /v1/delete",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets import datasets

        try:
            result = await datasets.delete_data(
                dataset_id=dataset_id,
                data_id=data_id,
                user=user,
                mode=mode,
            )
            return result

        except Exception as error:
            logger.error(f"Error during deletion by data_id: {str(error)}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
