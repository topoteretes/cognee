from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_verify_verify_api_v1_auth_verify_post import BodyVerifyVerifyApiV1AuthVerifyPost
from ...models.error_model import ErrorModel
from ...models.http_validation_error import HTTPValidationError
from ...models.user_read import UserRead
from ...types import Response


def _get_kwargs(
    *,
    body: BodyVerifyVerifyApiV1AuthVerifyPost,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/auth/verify",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[ErrorModel, HTTPValidationError, UserRead]]:
    if response.status_code == 200:
        response_200 = UserRead.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorModel.from_dict(response.json())

        return response_400

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[ErrorModel, HTTPValidationError, UserRead]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyVerifyVerifyApiV1AuthVerifyPost,
) -> Response[Union[ErrorModel, HTTPValidationError, UserRead]]:
    """Verify:Verify

    Args:
        body (BodyVerifyVerifyApiV1AuthVerifyPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ErrorModel, HTTPValidationError, UserRead]]
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
    client: Union[AuthenticatedClient, Client],
    body: BodyVerifyVerifyApiV1AuthVerifyPost,
) -> Optional[Union[ErrorModel, HTTPValidationError, UserRead]]:
    """Verify:Verify

    Args:
        body (BodyVerifyVerifyApiV1AuthVerifyPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ErrorModel, HTTPValidationError, UserRead]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyVerifyVerifyApiV1AuthVerifyPost,
) -> Response[Union[ErrorModel, HTTPValidationError, UserRead]]:
    """Verify:Verify

    Args:
        body (BodyVerifyVerifyApiV1AuthVerifyPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ErrorModel, HTTPValidationError, UserRead]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BodyVerifyVerifyApiV1AuthVerifyPost,
) -> Optional[Union[ErrorModel, HTTPValidationError, UserRead]]:
    """Verify:Verify

    Args:
        body (BodyVerifyVerifyApiV1AuthVerifyPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ErrorModel, HTTPValidationError, UserRead]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
