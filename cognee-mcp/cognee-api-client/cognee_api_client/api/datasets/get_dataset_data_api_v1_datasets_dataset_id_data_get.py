from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.data_dto import DataDTO
from ...models.error_response_dto import ErrorResponseDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    dataset_id: UUID,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/datasets/{dataset_id}/data",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[ErrorResponseDTO, HTTPValidationError, list["DataDTO"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = DataDTO.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 404:
        response_404 = ErrorResponseDTO.from_dict(response.json())

        return response_404

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[ErrorResponseDTO, HTTPValidationError, list["DataDTO"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    dataset_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[Union[ErrorResponseDTO, HTTPValidationError, list["DataDTO"]]]:
    """Get Dataset Data

     Get all data items in a dataset.

    This endpoint retrieves all data items (documents, files, etc.) that belong
    to a specific dataset. Each data item includes metadata such as name, type,
    creation time, and storage location.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns a list of data objects containing:
    - **id**: Unique data item identifier
    - **name**: Data item name
    - **created_at**: When the data was added
    - **updated_at**: When the data was last updated
    - **extension**: File extension
    - **mime_type**: MIME type of the data
    - **raw_data_location**: Storage location of the raw data

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ErrorResponseDTO, HTTPValidationError, list['DataDTO']]]
    """

    kwargs = _get_kwargs(
        dataset_id=dataset_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    dataset_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Optional[Union[ErrorResponseDTO, HTTPValidationError, list["DataDTO"]]]:
    """Get Dataset Data

     Get all data items in a dataset.

    This endpoint retrieves all data items (documents, files, etc.) that belong
    to a specific dataset. Each data item includes metadata such as name, type,
    creation time, and storage location.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns a list of data objects containing:
    - **id**: Unique data item identifier
    - **name**: Data item name
    - **created_at**: When the data was added
    - **updated_at**: When the data was last updated
    - **extension**: File extension
    - **mime_type**: MIME type of the data
    - **raw_data_location**: Storage location of the raw data

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ErrorResponseDTO, HTTPValidationError, list['DataDTO']]
    """

    return sync_detailed(
        dataset_id=dataset_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    dataset_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[Union[ErrorResponseDTO, HTTPValidationError, list["DataDTO"]]]:
    """Get Dataset Data

     Get all data items in a dataset.

    This endpoint retrieves all data items (documents, files, etc.) that belong
    to a specific dataset. Each data item includes metadata such as name, type,
    creation time, and storage location.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns a list of data objects containing:
    - **id**: Unique data item identifier
    - **name**: Data item name
    - **created_at**: When the data was added
    - **updated_at**: When the data was last updated
    - **extension**: File extension
    - **mime_type**: MIME type of the data
    - **raw_data_location**: Storage location of the raw data

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ErrorResponseDTO, HTTPValidationError, list['DataDTO']]]
    """

    kwargs = _get_kwargs(
        dataset_id=dataset_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    dataset_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Optional[Union[ErrorResponseDTO, HTTPValidationError, list["DataDTO"]]]:
    """Get Dataset Data

     Get all data items in a dataset.

    This endpoint retrieves all data items (documents, files, etc.) that belong
    to a specific dataset. Each data item includes metadata such as name, type,
    creation time, and storage location.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns a list of data objects containing:
    - **id**: Unique data item identifier
    - **name**: Data item name
    - **created_at**: When the data was added
    - **updated_at**: When the data was last updated
    - **extension**: File extension
    - **mime_type**: MIME type of the data
    - **raw_data_location**: Storage location of the raw data

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ErrorResponseDTO, HTTPValidationError, list['DataDTO']]
    """

    return (
        await asyncio_detailed(
            dataset_id=dataset_id,
            client=client,
        )
    ).parsed
