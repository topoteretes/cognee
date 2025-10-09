from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.dataset_creation_payload import DatasetCreationPayload
from ...models.dataset_dto import DatasetDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: DatasetCreationPayload,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/datasets",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[DatasetDTO, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = DatasetDTO.from_dict(response.json())

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
) -> Response[Union[DatasetDTO, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: DatasetCreationPayload,
) -> Response[Union[DatasetDTO, HTTPValidationError]]:
    """Create New Dataset

     Create a new dataset or return existing dataset with the same name.

    This endpoint creates a new dataset with the specified name. If a dataset
    with the same name already exists for the user, it returns the existing
    dataset instead of creating a duplicate. The user is automatically granted
    all permissions (read, write, share, delete) on the created dataset.

    ## Request Parameters
    - **dataset_data** (DatasetCreationPayload): Dataset creation parameters containing:
      - **name**: The name for the new dataset

    ## Response
    Returns the created or existing dataset object containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error creating dataset

    Args:
        body (DatasetCreationPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[DatasetDTO, HTTPValidationError]]
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
    body: DatasetCreationPayload,
) -> Optional[Union[DatasetDTO, HTTPValidationError]]:
    """Create New Dataset

     Create a new dataset or return existing dataset with the same name.

    This endpoint creates a new dataset with the specified name. If a dataset
    with the same name already exists for the user, it returns the existing
    dataset instead of creating a duplicate. The user is automatically granted
    all permissions (read, write, share, delete) on the created dataset.

    ## Request Parameters
    - **dataset_data** (DatasetCreationPayload): Dataset creation parameters containing:
      - **name**: The name for the new dataset

    ## Response
    Returns the created or existing dataset object containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error creating dataset

    Args:
        body (DatasetCreationPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[DatasetDTO, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: DatasetCreationPayload,
) -> Response[Union[DatasetDTO, HTTPValidationError]]:
    """Create New Dataset

     Create a new dataset or return existing dataset with the same name.

    This endpoint creates a new dataset with the specified name. If a dataset
    with the same name already exists for the user, it returns the existing
    dataset instead of creating a duplicate. The user is automatically granted
    all permissions (read, write, share, delete) on the created dataset.

    ## Request Parameters
    - **dataset_data** (DatasetCreationPayload): Dataset creation parameters containing:
      - **name**: The name for the new dataset

    ## Response
    Returns the created or existing dataset object containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error creating dataset

    Args:
        body (DatasetCreationPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[DatasetDTO, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: DatasetCreationPayload,
) -> Optional[Union[DatasetDTO, HTTPValidationError]]:
    """Create New Dataset

     Create a new dataset or return existing dataset with the same name.

    This endpoint creates a new dataset with the specified name. If a dataset
    with the same name already exists for the user, it returns the existing
    dataset instead of creating a duplicate. The user is automatically granted
    all permissions (read, write, share, delete) on the created dataset.

    ## Request Parameters
    - **dataset_data** (DatasetCreationPayload): Dataset creation parameters containing:
      - **name**: The name for the new dataset

    ## Response
    Returns the created or existing dataset object containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error creating dataset

    Args:
        body (DatasetCreationPayload):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[DatasetDTO, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
