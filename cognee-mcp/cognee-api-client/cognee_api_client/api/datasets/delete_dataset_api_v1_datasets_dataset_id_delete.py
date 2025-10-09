from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response_dto import ErrorResponseDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    dataset_id: UUID,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/api/v1/datasets/{dataset_id}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, ErrorResponseDTO, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Union[Any, ErrorResponseDTO, HTTPValidationError]]:
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
) -> Response[Union[Any, ErrorResponseDTO, HTTPValidationError]]:
    """Delete Dataset

     Delete a dataset by its ID.

    This endpoint permanently deletes a dataset and all its associated data.
    The user must have delete permissions on the dataset to perform this operation.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to delete

    ## Response
    No content returned on successful deletion.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error during deletion

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, ErrorResponseDTO, HTTPValidationError]]
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
) -> Optional[Union[Any, ErrorResponseDTO, HTTPValidationError]]:
    """Delete Dataset

     Delete a dataset by its ID.

    This endpoint permanently deletes a dataset and all its associated data.
    The user must have delete permissions on the dataset to perform this operation.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to delete

    ## Response
    No content returned on successful deletion.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error during deletion

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, ErrorResponseDTO, HTTPValidationError]
    """

    return sync_detailed(
        dataset_id=dataset_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    dataset_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[Union[Any, ErrorResponseDTO, HTTPValidationError]]:
    """Delete Dataset

     Delete a dataset by its ID.

    This endpoint permanently deletes a dataset and all its associated data.
    The user must have delete permissions on the dataset to perform this operation.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to delete

    ## Response
    No content returned on successful deletion.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error during deletion

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, ErrorResponseDTO, HTTPValidationError]]
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
) -> Optional[Union[Any, ErrorResponseDTO, HTTPValidationError]]:
    """Delete Dataset

     Delete a dataset by its ID.

    This endpoint permanently deletes a dataset and all its associated data.
    The user must have delete permissions on the dataset to perform this operation.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to delete

    ## Response
    No content returned on successful deletion.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error during deletion

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, ErrorResponseDTO, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            dataset_id=dataset_id,
            client=client,
        )
    ).parsed
