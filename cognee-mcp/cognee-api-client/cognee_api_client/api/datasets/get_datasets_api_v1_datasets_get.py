from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.dataset_dto import DatasetDTO
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/datasets",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[list["DatasetDTO"]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = DatasetDTO.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[list["DatasetDTO"]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[list["DatasetDTO"]]:
    """Get Datasets

     Get all datasets accessible to the authenticated user.

    This endpoint retrieves all datasets that the authenticated user has
    read permissions for. The datasets are returned with their metadata
    including ID, name, creation time, and owner information.

    ## Response
    Returns a list of dataset objects containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error retrieving datasets

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[list['DatasetDTO']]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
) -> Optional[list["DatasetDTO"]]:
    """Get Datasets

     Get all datasets accessible to the authenticated user.

    This endpoint retrieves all datasets that the authenticated user has
    read permissions for. The datasets are returned with their metadata
    including ID, name, creation time, and owner information.

    ## Response
    Returns a list of dataset objects containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error retrieving datasets

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        list['DatasetDTO']
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[list["DatasetDTO"]]:
    """Get Datasets

     Get all datasets accessible to the authenticated user.

    This endpoint retrieves all datasets that the authenticated user has
    read permissions for. The datasets are returned with their metadata
    including ID, name, creation time, and owner information.

    ## Response
    Returns a list of dataset objects containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error retrieving datasets

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[list['DatasetDTO']]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
) -> Optional[list["DatasetDTO"]]:
    """Get Datasets

     Get all datasets accessible to the authenticated user.

    This endpoint retrieves all datasets that the authenticated user has
    read permissions for. The datasets are returned with their metadata
    including ID, name, creation time, and owner information.

    ## Response
    Returns a list of dataset objects containing:
    - **id**: Unique dataset identifier
    - **name**: Dataset name
    - **created_at**: When the dataset was created
    - **updated_at**: When the dataset was last updated
    - **owner_id**: ID of the dataset owner

    ## Error Codes
    - **418 I'm a teapot**: Error retrieving datasets

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        list['DatasetDTO']
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
