from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    user_id: UUID,
    *,
    role_id: UUID,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_role_id = str(role_id)
    params["role_id"] = json_role_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/api/v1/permissions/users/{user_id}/roles",
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
    user_id: UUID,
    *,
    client: AuthenticatedClient,
    role_id: UUID,
) -> Response[Union[Any, HTTPValidationError]]:
    """Add User To Role

     Add a user to a role.

    This endpoint assigns a user to a specific role, granting them all the
    permissions associated with that role. The authenticated user must be
    the owner of the role or have appropriate administrative permissions.

    ## Path Parameters
    - **user_id** (UUID): The UUID of the user to add to the role

    ## Request Parameters
    - **role_id** (UUID): The UUID of the role to assign the user to

    ## Response
    Returns a success message indicating the user was added to the role.

    ## Error Codes
    - **400 Bad Request**: Invalid user or role ID
    - **403 Forbidden**: User doesn't have permission to assign roles
    - **404 Not Found**: User or role doesn't exist
    - **500 Internal Server Error**: Error adding user to role

    Args:
        user_id (UUID):
        role_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        role_id=role_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: UUID,
    *,
    client: AuthenticatedClient,
    role_id: UUID,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Add User To Role

     Add a user to a role.

    This endpoint assigns a user to a specific role, granting them all the
    permissions associated with that role. The authenticated user must be
    the owner of the role or have appropriate administrative permissions.

    ## Path Parameters
    - **user_id** (UUID): The UUID of the user to add to the role

    ## Request Parameters
    - **role_id** (UUID): The UUID of the role to assign the user to

    ## Response
    Returns a success message indicating the user was added to the role.

    ## Error Codes
    - **400 Bad Request**: Invalid user or role ID
    - **403 Forbidden**: User doesn't have permission to assign roles
    - **404 Not Found**: User or role doesn't exist
    - **500 Internal Server Error**: Error adding user to role

    Args:
        user_id (UUID):
        role_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        role_id=role_id,
    ).parsed


async def asyncio_detailed(
    user_id: UUID,
    *,
    client: AuthenticatedClient,
    role_id: UUID,
) -> Response[Union[Any, HTTPValidationError]]:
    """Add User To Role

     Add a user to a role.

    This endpoint assigns a user to a specific role, granting them all the
    permissions associated with that role. The authenticated user must be
    the owner of the role or have appropriate administrative permissions.

    ## Path Parameters
    - **user_id** (UUID): The UUID of the user to add to the role

    ## Request Parameters
    - **role_id** (UUID): The UUID of the role to assign the user to

    ## Response
    Returns a success message indicating the user was added to the role.

    ## Error Codes
    - **400 Bad Request**: Invalid user or role ID
    - **403 Forbidden**: User doesn't have permission to assign roles
    - **404 Not Found**: User or role doesn't exist
    - **500 Internal Server Error**: Error adding user to role

    Args:
        user_id (UUID):
        role_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        role_id=role_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: UUID,
    *,
    client: AuthenticatedClient,
    role_id: UUID,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Add User To Role

     Add a user to a role.

    This endpoint assigns a user to a specific role, granting them all the
    permissions associated with that role. The authenticated user must be
    the owner of the role or have appropriate administrative permissions.

    ## Path Parameters
    - **user_id** (UUID): The UUID of the user to add to the role

    ## Request Parameters
    - **role_id** (UUID): The UUID of the role to assign the user to

    ## Response
    Returns a success message indicating the user was added to the role.

    ## Error Codes
    - **400 Bad Request**: Invalid user or role ID
    - **403 Forbidden**: User doesn't have permission to assign roles
    - **404 Not Found**: User or role doesn't exist
    - **500 Internal Server Error**: Error adding user to role

    Args:
        user_id (UUID):
        role_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            role_id=role_id,
        )
    ).parsed
