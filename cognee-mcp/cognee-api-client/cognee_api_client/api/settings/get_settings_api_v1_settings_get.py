from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.settings_dto import SettingsDTO
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/settings",
    }

    return _kwargs


def _parse_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> Optional[SettingsDTO]:
    if response.status_code == 200:
        response_200 = SettingsDTO.from_dict(response.json())

        return response_200

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: Union[AuthenticatedClient, Client], response: httpx.Response) -> Response[SettingsDTO]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[SettingsDTO]:
    """Get Settings

     Get the current system settings.

    This endpoint retrieves the current configuration settings for the system,
    including LLM (Large Language Model) configuration and vector database
    configuration. These settings determine how the system processes and stores data.

    ## Response
    Returns the current system settings containing:
    - **llm**: LLM configuration (provider, model, API key)
    - **vector_db**: Vector database configuration (provider, URL, API key)

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving settings

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[SettingsDTO]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
) -> Optional[SettingsDTO]:
    """Get Settings

     Get the current system settings.

    This endpoint retrieves the current configuration settings for the system,
    including LLM (Large Language Model) configuration and vector database
    configuration. These settings determine how the system processes and stores data.

    ## Response
    Returns the current system settings containing:
    - **llm**: LLM configuration (provider, model, API key)
    - **vector_db**: Vector database configuration (provider, URL, API key)

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving settings

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        SettingsDTO
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
) -> Response[SettingsDTO]:
    """Get Settings

     Get the current system settings.

    This endpoint retrieves the current configuration settings for the system,
    including LLM (Large Language Model) configuration and vector database
    configuration. These settings determine how the system processes and stores data.

    ## Response
    Returns the current system settings containing:
    - **llm**: LLM configuration (provider, model, API key)
    - **vector_db**: Vector database configuration (provider, URL, API key)

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving settings

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[SettingsDTO]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
) -> Optional[SettingsDTO]:
    """Get Settings

     Get the current system settings.

    This endpoint retrieves the current configuration settings for the system,
    including LLM (Large Language Model) configuration and vector database
    configuration. These settings determine how the system processes and stores data.

    ## Response
    Returns the current system settings containing:
    - **llm**: LLM configuration (provider, model, API key)
    - **vector_db**: Vector database configuration (provider, URL, API key)

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving settings

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        SettingsDTO
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
