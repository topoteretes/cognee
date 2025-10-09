from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    data_id: UUID,
    dataset_id: UUID,
    mode: Union[Unset, str] = "soft",
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_data_id = str(data_id)
    params["data_id"] = json_data_id

    json_dataset_id = str(dataset_id)
    params["dataset_id"] = json_dataset_id

    params["mode"] = mode

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/api/v1/delete",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
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
    *,
    client: AuthenticatedClient,
    data_id: UUID,
    dataset_id: UUID,
    mode: Union[Unset, str] = "soft",
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Delete

     Delete data by its ID from the specified dataset.

    Args:
        data_id: The UUID of the data to delete
        dataset_id: The UUID of the dataset containing the data
        mode: \"soft\" (default) or \"hard\" - hard mode also deletes degree-one entity nodes
        user: Authenticated user

    Returns:
        JSON response indicating success or failure

    Args:
        data_id (UUID):
        dataset_id (UUID):
        mode (Union[Unset, str]):  Default: 'soft'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        data_id=data_id,
        dataset_id=dataset_id,
        mode=mode,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    data_id: UUID,
    dataset_id: UUID,
    mode: Union[Unset, str] = "soft",
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Delete

     Delete data by its ID from the specified dataset.

    Args:
        data_id: The UUID of the data to delete
        dataset_id: The UUID of the dataset containing the data
        mode: \"soft\" (default) or \"hard\" - hard mode also deletes degree-one entity nodes
        user: Authenticated user

    Returns:
        JSON response indicating success or failure

    Args:
        data_id (UUID):
        dataset_id (UUID):
        mode (Union[Unset, str]):  Default: 'soft'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        data_id=data_id,
        dataset_id=dataset_id,
        mode=mode,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    data_id: UUID,
    dataset_id: UUID,
    mode: Union[Unset, str] = "soft",
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Delete

     Delete data by its ID from the specified dataset.

    Args:
        data_id: The UUID of the data to delete
        dataset_id: The UUID of the dataset containing the data
        mode: \"soft\" (default) or \"hard\" - hard mode also deletes degree-one entity nodes
        user: Authenticated user

    Returns:
        JSON response indicating success or failure

    Args:
        data_id (UUID):
        dataset_id (UUID):
        mode (Union[Unset, str]):  Default: 'soft'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        data_id=data_id,
        dataset_id=dataset_id,
        mode=mode,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    data_id: UUID,
    dataset_id: UUID,
    mode: Union[Unset, str] = "soft",
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Delete

     Delete data by its ID from the specified dataset.

    Args:
        data_id: The UUID of the data to delete
        dataset_id: The UUID of the dataset containing the data
        mode: \"soft\" (default) or \"hard\" - hard mode also deletes degree-one entity nodes
        user: Authenticated user

    Returns:
        JSON response indicating success or failure

    Args:
        data_id (UUID):
        dataset_id (UUID):
        mode (Union[Unset, str]):  Default: 'soft'.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            data_id=data_id,
            dataset_id=dataset_id,
            mode=mode,
        )
    ).parsed
