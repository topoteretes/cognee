from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.search_history_item import SearchHistoryItem
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/search",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[list["SearchHistoryItem"]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = SearchHistoryItem.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[list["SearchHistoryItem"]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[list["SearchHistoryItem"]]:
    """Get Search History

     Get search history for the authenticated user.

    This endpoint retrieves the search history for the authenticated user,
    returning a list of previously executed searches with their timestamps.

    ## Response
    Returns a list of search history items containing:
    - **id**: Unique identifier for the search
    - **text**: The search query text
    - **user**: User who performed the search
    - **created_at**: When the search was performed

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving search history

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[list['SearchHistoryItem']]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
) -> Optional[list["SearchHistoryItem"]]:
    """Get Search History

     Get search history for the authenticated user.

    This endpoint retrieves the search history for the authenticated user,
    returning a list of previously executed searches with their timestamps.

    ## Response
    Returns a list of search history items containing:
    - **id**: Unique identifier for the search
    - **text**: The search query text
    - **user**: User who performed the search
    - **created_at**: When the search was performed

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving search history

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        list['SearchHistoryItem']
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[list["SearchHistoryItem"]]:
    """Get Search History

     Get search history for the authenticated user.

    This endpoint retrieves the search history for the authenticated user,
    returning a list of previously executed searches with their timestamps.

    ## Response
    Returns a list of search history items containing:
    - **id**: Unique identifier for the search
    - **text**: The search query text
    - **user**: User who performed the search
    - **created_at**: When the search was performed

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving search history

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[list['SearchHistoryItem']]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
) -> Optional[list["SearchHistoryItem"]]:
    """Get Search History

     Get search history for the authenticated user.

    This endpoint retrieves the search history for the authenticated user,
    returning a list of previously executed searches with their timestamps.

    ## Response
    Returns a list of search history items containing:
    - **id**: Unique identifier for the search
    - **text**: The search query text
    - **user**: User who performed the search
    - **created_at**: When the search was performed

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving search history

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        list['SearchHistoryItem']
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
