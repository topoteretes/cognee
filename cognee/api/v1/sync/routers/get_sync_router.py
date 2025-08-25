from uuid import UUID

from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse


from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.api.v1.sync import SyncResponse

logger = get_logger()


def get_sync_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=SyncResponse)
    async def sync_to_cloud(
        dataset_id: UUID = Form(...),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Sync local data to Cognee Cloud.

        This endpoint triggers synchronization of local Cognee data to your cloud instance.
        It uploads your local datasets, knowledge graphs, and processed data to the cloud
        for backup, sharing, or cloud-based processing.

        ## Request Parameters
        - **dataset_id** (UUID): UUID of the dataset to sync to cloud

        ## Response
        Returns immediate response for the sync operation:
        - **run_id**: Unique identifier for tracking the background sync operation
        - **status**: Always "started" (operation runs in background)
        - **dataset_id**: ID of the dataset being synced
        - **dataset_name**: Name of the dataset being synced
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
        # Sync dataset to cloud by ID
        curl -X POST "http://localhost:8000/api/v1/sync" \\
          -H "Authorization: Bearer your-token" \\
          -F "dataset_id=123e4567-e89b-12d3-a456-426614174000"
        ```

        ## Error Codes
        - **400 Bad Request**: Invalid dataset_id format
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
                "dataset_id": str(dataset_id),
            },
        )

        from cognee.api.v1.sync import sync as cognee_sync

        try:
            # Retrieve existing dataset and check permissions
            dataset = await get_specific_user_permission_datasets(user.id, "write", [dataset_id])
            
            # Convert from list to Dataset element
            if isinstance(dataset, list):
                if not dataset:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Dataset with ID {dataset_id} not found or access denied."}
                    )
                dataset = dataset[0]

            await set_database_global_context_variables(dataset.id, dataset.owner_id)

            # Execute cloud sync operation
            sync_result = await cognee_sync(
                dataset=dataset,
                user=user,
            )

            return sync_result

        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except PermissionError as e:
            return JSONResponse(status_code=403, content={"error": str(e)})
        except ConnectionError as e:
            return JSONResponse(status_code=409, content={"error": f"Cloud service unavailable: {str(e)}"})
        except Exception as e:
            logger.error(f"Cloud sync operation failed: {str(e)}")
            return JSONResponse(status_code=409, content={"error": f"Cloud sync operation failed: {str(e)}"})

    return router
