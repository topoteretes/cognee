from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from cognee.api.v1.models import InDTO
from cognee.api.v1.save.save import save as save_fn
from cognee.modules.users.models import User
from cognee.api.v1.auth.dependencies.get_authenticated_user import get_authenticated_user
from cognee.shared.telemetry import send_telemetry


class SavePayloadDTO(InDTO):
    datasets: Optional[List[str]] = None
    dataset_ids: Optional[List[UUID]] = None
    export_root_directory: Optional[str] = None
    # alias support
    path: Optional[str] = None


def get_save_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def save(payload: SavePayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Save dataset exports to markdown files.

        For each accessible dataset, produces a folder with one markdown per data item
        containing summary, path ascii tree, question ideas, and search results.
        """
        send_telemetry(
            "Save API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/save",
            },
        )

        try:
            datasets = payload.datasets if payload.datasets else payload.dataset_ids
            result = await save_fn(
                datasets=datasets,
                export_root_directory=payload.export_root_directory or payload.path,
                user=user,
            )
            return result
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Error during save operation: {str(error)}",
            ) from error

    return router
