from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.sync_request import SyncRequest
from ...models.sync_to_cloud_api_v1_sync_post_response_sync_to_cloud_api_v1_sync_post import (
    SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost,
)
from ...types import Response


def _get_kwargs(
    *,
    body: SyncRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/sync",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]:
    if response.status_code == 200:
        response_200 = SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost.from_dict(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: SyncRequest,
) -> Response[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]:
    r""" Sync To Cloud

     Sync local data to Cognee Cloud.

    This endpoint triggers synchronization of local Cognee data to your cloud instance.
    It uploads your local datasets, knowledge graphs, and processed data to the cloud
    for backup, sharing, or cloud-based processing.

    ## Request Body (JSON)
    ```json
    {
        \"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]
    }
    ```

    ## Response
    Returns immediate response for the sync operation:
    - **run_id**: Unique identifier for tracking the background sync operation
    - **status**: Always \"started\" (operation runs in background)
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
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
      -d '{\"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]}'

    # Sync all user datasets (empty request body or null dataset_ids)
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
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

    Args:
        body (SyncRequest): Request model for sync operations.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]
     """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: SyncRequest,
) -> Optional[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]:
    r""" Sync To Cloud

     Sync local data to Cognee Cloud.

    This endpoint triggers synchronization of local Cognee data to your cloud instance.
    It uploads your local datasets, knowledge graphs, and processed data to the cloud
    for backup, sharing, or cloud-based processing.

    ## Request Body (JSON)
    ```json
    {
        \"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]
    }
    ```

    ## Response
    Returns immediate response for the sync operation:
    - **run_id**: Unique identifier for tracking the background sync operation
    - **status**: Always \"started\" (operation runs in background)
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
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
      -d '{\"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]}'

    # Sync all user datasets (empty request body or null dataset_ids)
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
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

    Args:
        body (SyncRequest): Request model for sync operations.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]
     """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: SyncRequest,
) -> Response[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]:
    r""" Sync To Cloud

     Sync local data to Cognee Cloud.

    This endpoint triggers synchronization of local Cognee data to your cloud instance.
    It uploads your local datasets, knowledge graphs, and processed data to the cloud
    for backup, sharing, or cloud-based processing.

    ## Request Body (JSON)
    ```json
    {
        \"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]
    }
    ```

    ## Response
    Returns immediate response for the sync operation:
    - **run_id**: Unique identifier for tracking the background sync operation
    - **status**: Always \"started\" (operation runs in background)
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
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
      -d '{\"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]}'

    # Sync all user datasets (empty request body or null dataset_ids)
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
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

    Args:
        body (SyncRequest): Request model for sync operations.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]
     """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: SyncRequest,
) -> Optional[Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]]:
    r""" Sync To Cloud

     Sync local data to Cognee Cloud.

    This endpoint triggers synchronization of local Cognee data to your cloud instance.
    It uploads your local datasets, knowledge graphs, and processed data to the cloud
    for backup, sharing, or cloud-based processing.

    ## Request Body (JSON)
    ```json
    {
        \"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]
    }
    ```

    ## Response
    Returns immediate response for the sync operation:
    - **run_id**: Unique identifier for tracking the background sync operation
    - **status**: Always \"started\" (operation runs in background)
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
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
      -d '{\"dataset_ids\": [\"123e4567-e89b-12d3-a456-426614174000\",
    \"456e7890-e12b-34c5-d678-901234567000\"]}'

    # Sync all user datasets (empty request body or null dataset_ids)
    curl -X POST \"http://localhost:8000/api/v1/sync\" \
      -H \"Content-Type: application/json\" \
      -H \"Cookie: auth_token=your-token\" \
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

    Args:
        body (SyncRequest): Request model for sync operations.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SyncToCloudApiV1SyncPostResponseSyncToCloudApiV1SyncPost]
     """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
