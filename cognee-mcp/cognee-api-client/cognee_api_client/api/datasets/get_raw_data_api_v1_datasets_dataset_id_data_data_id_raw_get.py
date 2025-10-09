from http import HTTPStatus
from typing import Any, Optional, Union, cast
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    dataset_id: UUID,
    data_id: UUID,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = cast(Any, None)
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
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    dataset_id: UUID,
    data_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[Union[Any, HTTPValidationError]]:
    """Get Raw Data

     Download the raw data file for a specific data item.

    This endpoint allows users to download the original, unprocessed data file
    for a specific data item within a dataset. The file is returned as a direct
    download with appropriate headers.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset containing the data
    - **data_id** (UUID): The unique identifier of the data item to download

    ## Response
    Returns the raw data file as a downloadable response.

    ## Error Codes
    - **404 Not Found**: Dataset or data item doesn't exist, or user doesn't have access
    - **500 Internal Server Error**: Error accessing the raw data file

    Args:
        dataset_id (UUID):
        data_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        dataset_id=dataset_id,
        data_id=data_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    dataset_id: UUID,
    data_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Get Raw Data

     Download the raw data file for a specific data item.

    This endpoint allows users to download the original, unprocessed data file
    for a specific data item within a dataset. The file is returned as a direct
    download with appropriate headers.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset containing the data
    - **data_id** (UUID): The unique identifier of the data item to download

    ## Response
    Returns the raw data file as a downloadable response.

    ## Error Codes
    - **404 Not Found**: Dataset or data item doesn't exist, or user doesn't have access
    - **500 Internal Server Error**: Error accessing the raw data file

    Args:
        dataset_id (UUID):
        data_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        dataset_id=dataset_id,
        data_id=data_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    dataset_id: UUID,
    data_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[Union[Any, HTTPValidationError]]:
    """Get Raw Data

     Download the raw data file for a specific data item.

    This endpoint allows users to download the original, unprocessed data file
    for a specific data item within a dataset. The file is returned as a direct
    download with appropriate headers.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset containing the data
    - **data_id** (UUID): The unique identifier of the data item to download

    ## Response
    Returns the raw data file as a downloadable response.

    ## Error Codes
    - **404 Not Found**: Dataset or data item doesn't exist, or user doesn't have access
    - **500 Internal Server Error**: Error accessing the raw data file

    Args:
        dataset_id (UUID):
        data_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        dataset_id=dataset_id,
        data_id=data_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    dataset_id: UUID,
    data_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Get Raw Data

     Download the raw data file for a specific data item.

    This endpoint allows users to download the original, unprocessed data file
    for a specific data item within a dataset. The file is returned as a direct
    download with appropriate headers.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset containing the data
    - **data_id** (UUID): The unique identifier of the data item to download

    ## Response
    Returns the raw data file as a downloadable response.

    ## Error Codes
    - **404 Not Found**: Dataset or data item doesn't exist, or user doesn't have access
    - **500 Internal Server Error**: Error accessing the raw data file

    Args:
        dataset_id (UUID):
        data_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            dataset_id=dataset_id,
            data_id=data_id,
            client=client,
        )
    ).parsed
