from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/sync/status",
    }

    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> Optional[Any]:
    if response.status_code == 200:
        return None

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[Any]:
    r""" Get Sync Status Overview

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
    curl -X GET \"http://localhost:8000/api/v1/sync/status\" \
      -H \"Cookie: auth_token=your-token\"
    ```

    ## Example Responses

    **No running syncs:**
    ```json
    {
      \"has_running_sync\": false,
      \"running_sync_count\": 0
    }
    ```

    **With running sync:**
    ```json
    {
      \"has_running_sync\": true,
      \"running_sync_count\": 1,
      \"latest_running_sync\": {
        \"run_id\": \"12345678-1234-5678-9012-123456789012\",
        \"dataset_name\": \"My Dataset\",
        \"progress_percentage\": 45,
        \"created_at\": \"2025-01-01T00:00:00Z\"
      }
    }
    ```

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
     """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[Any]:
    r""" Get Sync Status Overview

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
    curl -X GET \"http://localhost:8000/api/v1/sync/status\" \
      -H \"Cookie: auth_token=your-token\"
    ```

    ## Example Responses

    **No running syncs:**
    ```json
    {
      \"has_running_sync\": false,
      \"running_sync_count\": 0
    }
    ```

    **With running sync:**
    ```json
    {
      \"has_running_sync\": true,
      \"running_sync_count\": 1,
      \"latest_running_sync\": {
        \"run_id\": \"12345678-1234-5678-9012-123456789012\",
        \"dataset_name\": \"My Dataset\",
        \"progress_percentage\": 45,
        \"created_at\": \"2025-01-01T00:00:00Z\"
      }
    }
    ```

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
     """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)
