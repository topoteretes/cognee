from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    principal_id: UUID,
    *,
    body: list[UUID],
    permission_name: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["permission_name"] = permission_name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/api/v1/permissions/datasets/{principal_id}",
        "params": params,
    }

    _kwargs["json"] = []
    for body_item_data in body:
        body_item = str(body_item_data)
        _kwargs["json"].append(body_item)

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
    principal_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[UUID],
    permission_name: str,
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Give Datasets Permission To Principal

     Grant permission on datasets to a principal (user or role).

    This endpoint allows granting specific permissions on one or more datasets
    to a principal (which can be a user or role). The authenticated user must
    have appropriate permissions to grant access to the specified datasets.

    ## Path Parameters
    - **principal_id** (UUID): The UUID of the principal (user or role) to grant permission to

    ## Request Parameters
    - **permission_name** (str): The name of the permission to grant (e.g., \"read\", \"write\",
    \"delete\")
    - **dataset_ids** (List[UUID]): List of dataset UUIDs to grant permission on

    ## Response
    Returns a success message indicating permission was assigned.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **403 Forbidden**: User doesn't have permission to grant access
    - **500 Internal Server Error**: Error granting permission

    Args:
        principal_id (UUID):
        permission_name (str):
        body (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        principal_id=principal_id,
        body=body,
        permission_name=permission_name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    principal_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[UUID],
    permission_name: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Give Datasets Permission To Principal

     Grant permission on datasets to a principal (user or role).

    This endpoint allows granting specific permissions on one or more datasets
    to a principal (which can be a user or role). The authenticated user must
    have appropriate permissions to grant access to the specified datasets.

    ## Path Parameters
    - **principal_id** (UUID): The UUID of the principal (user or role) to grant permission to

    ## Request Parameters
    - **permission_name** (str): The name of the permission to grant (e.g., \"read\", \"write\",
    \"delete\")
    - **dataset_ids** (List[UUID]): List of dataset UUIDs to grant permission on

    ## Response
    Returns a success message indicating permission was assigned.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **403 Forbidden**: User doesn't have permission to grant access
    - **500 Internal Server Error**: Error granting permission

    Args:
        principal_id (UUID):
        permission_name (str):
        body (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        principal_id=principal_id,
        client=client,
        body=body,
        permission_name=permission_name,
    ).parsed


async def asyncio_detailed(
    principal_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[UUID],
    permission_name: str,
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Give Datasets Permission To Principal

     Grant permission on datasets to a principal (user or role).

    This endpoint allows granting specific permissions on one or more datasets
    to a principal (which can be a user or role). The authenticated user must
    have appropriate permissions to grant access to the specified datasets.

    ## Path Parameters
    - **principal_id** (UUID): The UUID of the principal (user or role) to grant permission to

    ## Request Parameters
    - **permission_name** (str): The name of the permission to grant (e.g., \"read\", \"write\",
    \"delete\")
    - **dataset_ids** (List[UUID]): List of dataset UUIDs to grant permission on

    ## Response
    Returns a success message indicating permission was assigned.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **403 Forbidden**: User doesn't have permission to grant access
    - **500 Internal Server Error**: Error granting permission

    Args:
        principal_id (UUID):
        permission_name (str):
        body (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        principal_id=principal_id,
        body=body,
        permission_name=permission_name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    principal_id: UUID,
    *,
    client: AuthenticatedClient,
    body: list[UUID],
    permission_name: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Give Datasets Permission To Principal

     Grant permission on datasets to a principal (user or role).

    This endpoint allows granting specific permissions on one or more datasets
    to a principal (which can be a user or role). The authenticated user must
    have appropriate permissions to grant access to the specified datasets.

    ## Path Parameters
    - **principal_id** (UUID): The UUID of the principal (user or role) to grant permission to

    ## Request Parameters
    - **permission_name** (str): The name of the permission to grant (e.g., \"read\", \"write\",
    \"delete\")
    - **dataset_ids** (List[UUID]): List of dataset UUIDs to grant permission on

    ## Response
    Returns a success message indicating permission was assigned.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **403 Forbidden**: User doesn't have permission to grant access
    - **500 Internal Server Error**: Error granting permission

    Args:
        principal_id (UUID):
        permission_name (str):
        body (list[UUID]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            principal_id=principal_id,
            client=client,
            body=body,
            permission_name=permission_name,
        )
    ).parsed
