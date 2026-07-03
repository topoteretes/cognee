"""FastAPI router for dataset versioning and snapshot management.

Exposes endpoints to create, list, and delete snapshots of a dataset, as well as
undoing soft-deleted data items.
"""

from typing import Dict, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from cognee.infrastructure.databases.versioning.models import SnapshotPointer
from cognee.infrastructure.databases.versioning.version_manager import get_version_manager
from cognee.modules.users.methods import get_default_user

router = APIRouter(prefix="/datasets", tags=["versioning"])


class CreateSnapshotRequest(BaseModel):
    """Request schema for creating a snapshot."""

    name: str = Field(..., description="The user-defined name of the snapshot.")


class UndoForgetRequest(BaseModel):
    """Request schema for undoing a forget operation."""

    data_id: UUID = Field(..., description="The UUID of the soft-deleted data item to restore.")


@router.post("/{dataset_id}/snapshots", response_model=SnapshotPointer, status_code=status.HTTP_201_CREATED)
async def create_snapshot(
    dataset_id: UUID,
    request: CreateSnapshotRequest,
    user=Depends(get_default_user),
) -> SnapshotPointer:
    """Create a new named snapshot of the current version of the dataset.

    Args:
        dataset_id: The UUID of the dataset to snapshot.
        request: The request body containing the snapshot name.
        user: The authenticated user context.

    Returns:
        SnapshotPointer describing the created snapshot.
    """
    version_manager = get_version_manager(dataset_id)
    # Get current version ID.
    current_version = await version_manager.get_current_version()

    try:
        snapshot = await version_manager.snapshot_store.create_snapshot(
            name=request.name,
            dataset_id=dataset_id,
            version_id=current_version,
        )
        return snapshot
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


@router.get("/{dataset_id}/snapshots", response_model=List[SnapshotPointer])
async def list_snapshots(
    dataset_id: UUID,
    user=Depends(get_default_user),
) -> List[SnapshotPointer]:
    """Retrieve all snapshot pointers registered for the specified dataset.

    Args:
        dataset_id: The UUID of the dataset.
        user: The authenticated user context.

    Returns:
        A list of SnapshotPointer records.
    """
    version_manager = get_version_manager(dataset_id)
    return await version_manager.snapshot_store.list_snapshots(dataset_id)


@router.delete("/{dataset_id}/snapshots/{name}", response_model=Dict[str, bool])
async def delete_snapshot(
    dataset_id: UUID,
    name: str,
    user=Depends(get_default_user),
) -> Dict[str, bool]:
    """Delete a named snapshot pointer from the dataset.

    Args:
        dataset_id: The UUID of the dataset.
        name: The name of the snapshot.
        user: The authenticated user context.

    Returns:
        A confirmation dict indicating success.
    """
    version_manager = get_version_manager(dataset_id)
    try:
        await version_manager.snapshot_store.delete_snapshot(name, dataset_id)
        return {"deleted": True}
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )


@router.post("/{dataset_id}/forget/undo", response_model=Dict[str, int])
async def undo_forget_endpoint(
    dataset_id: UUID,
    request: UndoForgetRequest,
    user=Depends(get_default_user),
) -> Dict[str, int]:
    """Undo a soft-delete (forget) operation for a specific data item.

    Restores all previously tombstoned graph nodes and returns the count.

    Args:
        dataset_id: The UUID of the dataset.
        request: The request body containing the data item ID.
        user: The authenticated user.

    Returns:
        A dictionary containing the count of restored items.
    """
    from cognee.api.v1.forget.forget import forget

    # Call forget with undo=True to restore the items.
    result = await forget(
        data_id=request.data_id,
        dataset_id=dataset_id,
        user=user,
        undo=True,
    )
    restored_count = result.get("restored_count", 0)
    return {"restored_count": restored_count}
