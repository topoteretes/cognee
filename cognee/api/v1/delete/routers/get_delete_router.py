from fastapi import Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_delete_router() -> APIRouter:
    router = APIRouter()

    @router.delete("", response_model=None)
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
        from cognee.api.v1.delete import delete as cognee_delete

        try:
            result = await cognee_delete(
                data_id=data_id,
                dataset_id=dataset_id,
                mode=mode,
                user=user,
            )
            return result

        except Exception as error:
            logger.error(f"Error during deletion by data_id: {str(error)}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
