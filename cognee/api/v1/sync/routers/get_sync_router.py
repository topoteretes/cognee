import os
from uuid import UUID
from typing import Optional, Dict, Any, Union, Literal

from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_sync_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def sync_to_cloud(
        dataset_name: Optional[str] = Form(default=None),
        dataset_id: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Sync local data to Cognee Cloud.

        This endpoint triggers synchronization of local Cognee data to your cloud instance.
        It uploads your local datasets, knowledge graphs, and processed data to the cloud
        for backup, sharing, or cloud-based processing.

        ## Request Parameters
        - **dataset_name** (Optional[str]): Name of the local dataset to sync to cloud
        - **dataset_id** (Optional[UUID]): UUID of specific dataset to sync to cloud

        Either dataset_name or dataset_id must be provided.

        ## Response
        Returns information about the sync operation:
        - **sync_id**: Unique identifier for tracking the sync operation
        - **status**: Current status ("started", "completed", "failed")
        - **source_info**: Information about local data being synced
        - **records_processed**: Number of records synchronized to cloud
        - **bytes_transferred**: Amount of data uploaded to cloud
        - **errors**: List of any errors encountered
        - **timestamp**: When the sync was initiated
        - **duration**: How long the sync operation took

        ## Cloud Sync Features
        - **Automatic Authentication**: Uses your Cognee Cloud credentials
        - **Data Compression**: Optimizes transfer size for faster uploads
        - **Smart Sync**: Automatically handles data updates efficiently
        - **Progress Tracking**: Monitor sync status with sync_id
        - **Error Recovery**: Automatic retry for failed transfers
        - **Data Validation**: Ensures data integrity during transfer

        ## Example Usage
        ```bash
        # Sync specific dataset to cloud
        curl -X POST "http://localhost:8000/api/v1/sync" \\
          -H "Authorization: Bearer your-token" \\
          -F "dataset_name=main_dataset"

        # Sync by dataset ID
        curl -X POST "http://localhost:8000/api/v1/sync" \\
          -H "Authorization: Bearer your-token" \\
          -F "dataset_id=123e4567-e89b-12d3-a456-426614174000"
        ```

        ## Error Codes
        - **400 Bad Request**: Missing dataset_name or dataset_id
        - **401 Unauthorized**: Invalid or missing authentication
        - **403 Forbidden**: User doesn't have permission to access dataset
        - **409 Conflict**: Sync operation conflict or cloud service unavailable
        - **413 Payload Too Large**: Dataset too large for current cloud plan
        - **429 Too Many Requests**: Rate limit exceeded

        ## Notes
        - Sync operations are asynchronous for large datasets
        - Progress can be tracked using the returned sync_id
        - Large datasets are automatically chunked for efficient transfer
        - Cloud storage usage counts against your plan limits
        """
        send_telemetry(
            "Cloud Sync API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/sync",
                "dataset_name": dataset_name,
                "dataset_id": str(dataset_id) if dataset_id else None,
            },
        )

        from cognee.api.v1.sync import sync as cognee_sync

        if not dataset_id and not dataset_name:
            return JSONResponse(
                status_code=400,
                content={"error": "Either dataset_id or dataset_name must be provided."}
            )

        try:
            # Execute cloud sync operation
            sync_result = await cognee_sync(
                source=f"dataset:{dataset_name or str(dataset_id)}",
                user=user,
                dataset_id=dataset_id,
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
