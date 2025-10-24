from uuid import UUID
from typing import Optional, List
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse


from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from cognee.modules.sync.methods import get_running_sync_operations_for_user, get_sync_operation
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.sync import SyncResponse
from cognee import __version__ as cognee_version
from cognee.context_global_variables import set_database_global_context_variables

logger = get_logger()


class SyncRequest(InDTO):
    """Request model for sync operations."""

    dataset_ids: Optional[List[UUID]] = None


def get_sync_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict[str, SyncResponse])
    async def sync_to_cloud(
        request: SyncRequest,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Sync local data to Cognee Cloud.

        This endpoint triggers synchronization of local Cognee data to your cloud instance.
        It uploads your local datasets, knowledge graphs, and processed data to the cloud
        for backup, sharing, or cloud-based processing.

        ## Request Body (JSON)
        ```json
        {
            "dataset_ids": ["123e4567-e89b-12d3-a456-426614174000", "456e7890-e12b-34c5-d678-901234567000"]
        }
        ```

        ## Response
        Returns immediate response for the sync operation:
        - **run_id**: Unique identifier for tracking the background sync operation
        - **status**: Always "started" (operation runs in background)
        - **dataset_ids**: List of dataset IDs being synced
        - **dataset_names**: List of dataset names being synced
        - **message**: Description of the background operation
        - **timestamp**: When the sync was initiated
        - **user_id**: User who initiated the sync

        ## Cloud Sync Features
        - **Automatic Authentication**: Uses your Cognee Cloud credentials
        - **Data Compression**: Optimizes transfer size for faster uploads
        - **Smart Sync**: Automatically handles data updates efficiently
        - **Progress Tracking**: Monitor sync status with sync_id
        - **Error Recovery**: Automatic retry for failed transfers
        - **Data Validation**: Ensures data integrity during transfer

        ## Example Usage
        ```bash
        # Sync multiple datasets to cloud by IDs (JSON request)
        curl -X POST "http://localhost:8000/api/v1/sync" \\
          -H "Content-Type: application/json" \\
          -H "Cookie: auth_token=your-token" \\
          -d '{"dataset_ids": ["123e4567-e89b-12d3-a456-426614174000", "456e7890-e12b-34c5-d678-901234567000"]}'
        
        # Sync all user datasets (empty request body or null dataset_ids)
        curl -X POST "http://localhost:8000/api/v1/sync" \\
          -H "Content-Type: application/json" \\
          -H "Cookie: auth_token=your-token" \\
          -d '{}'
        ```

        ## Error Codes
        - **400 Bad Request**: Invalid dataset_ids format
        - **401 Unauthorized**: Invalid or missing authentication
        - **403 Forbidden**: User doesn't have permission to access dataset
        - **404 Not Found**: Dataset not found
        - **409 Conflict**: Sync operation conflict or cloud service unavailable
        - **413 Payload Too Large**: Dataset too large for current cloud plan
        - **429 Too Many Requests**: Rate limit exceeded

        ## Notes  
        - Sync operations run in the background - you get an immediate response
        - Use the returned run_id to track progress (status API coming soon)
        - Large datasets are automatically chunked for efficient transfer
        - Cloud storage usage counts against your plan limits
        - The sync will continue even if you close your connection
        """
        send_telemetry(
            "Cloud Sync API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/sync",
                "cognee_version": cognee_version,
                "dataset_ids": [str(id) for id in request.dataset_ids]
                if request.dataset_ids
                else "*",
            },
        )

        from cognee.api.v1.sync import sync as cognee_sync

        try:
            # Check if user has any running sync operations
            running_syncs = await get_running_sync_operations_for_user(user.id)
            if running_syncs:
                # Return information about the existing sync operation
                existing_sync = running_syncs[0]  # Get the most recent running sync
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "Sync operation already in progress",
                        "details": {
                            "run_id": existing_sync.run_id,
                            "status": "already_running",
                            "dataset_ids": existing_sync.dataset_ids,
                            "dataset_names": existing_sync.dataset_names,
                            "message": f"You have a sync operation already in progress with run_id '{existing_sync.run_id}'. Use the status endpoint to monitor progress, or wait for it to complete before starting a new sync.",
                            "timestamp": existing_sync.created_at.isoformat(),
                            "progress_percentage": existing_sync.progress_percentage,
                        },
                    },
                )

            # Retrieve existing dataset and check permissions
            datasets = await get_specific_user_permission_datasets(
                user.id, "write", request.dataset_ids if request.dataset_ids else None
            )

            # Execute new cloud sync operation for all datasets
            sync_result = await cognee_sync(
                datasets=datasets,
                user=user,
            )

            return sync_result

        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except PermissionError as e:
            return JSONResponse(status_code=403, content={"error": str(e)})
        except ConnectionError as e:
            return JSONResponse(
                status_code=409, content={"error": f"Cloud service unavailable: {str(e)}"}
            )
        except Exception as e:
            logger.error(f"Cloud sync operation failed: {str(e)}")
            return JSONResponse(status_code=409, content={"error": "Cloud sync operation failed."})

    @router.get("/status")
    async def get_sync_status_overview(
        user: User = Depends(get_authenticated_user),
    ):
        """
        Check if there are any running sync operations for the current user.

        This endpoint provides a simple check to see if the user has any active sync operations
        without needing to know specific run IDs.

        ## Response
        Returns a simple status overview:
        - **has_running_sync**: Boolean indicating if there are any running syncs
        - **running_sync_count**: Number of currently running sync operations
        - **latest_running_sync** (optional): Information about the most recent running sync if any exists

        ## Example Usage
        ```bash
        curl -X GET "http://localhost:8000/api/v1/sync/status" \\
          -H "Cookie: auth_token=your-token"
        ```

        ## Example Responses
        
        **No running syncs:**
        ```json
        {
          "has_running_sync": false,
          "running_sync_count": 0
        }
        ```

        **With running sync:**
        ```json
        {
          "has_running_sync": true,
          "running_sync_count": 1,
          "latest_running_sync": {
            "run_id": "12345678-1234-5678-9012-123456789012",
            "dataset_name": "My Dataset",
            "progress_percentage": 45,
            "created_at": "2025-01-01T00:00:00Z"
          }
        }
        ```
        """
        send_telemetry(
            "Sync Status Overview API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/sync/status",
                "cognee_version": cognee_version,
            },
        )

        try:
            # Get any running sync operations for the user
            running_syncs = await get_running_sync_operations_for_user(user.id)

            response = {
                "has_running_sync": len(running_syncs) > 0,
                "running_sync_count": len(running_syncs),
            }

            # If there are running syncs, include info about the latest one
            if running_syncs:
                latest_sync = running_syncs[0]  # Already ordered by created_at desc
                response["latest_running_sync"] = {
                    "run_id": latest_sync.run_id,
                    "dataset_ids": latest_sync.dataset_ids,
                    "dataset_names": latest_sync.dataset_names,
                    "progress_percentage": latest_sync.progress_percentage,
                    "created_at": latest_sync.created_at.isoformat()
                    if latest_sync.created_at
                    else None,
                }

            return response

        except Exception as e:
            logger.error(f"Failed to get sync status overview: {str(e)}")
            return JSONResponse(
                status_code=500, content={"error": "Failed to get sync status overview"}
            )

    return router
