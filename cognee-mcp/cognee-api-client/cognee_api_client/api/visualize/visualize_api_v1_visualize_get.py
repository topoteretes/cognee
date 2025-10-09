from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    dataset_id: UUID,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_dataset_id = str(dataset_id)
    params["dataset_id"] = json_dataset_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/visualize",
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
    dataset_id: UUID,
) -> Response[Union[Any, HTTPValidationError]]:
    """Visualize

     Generate an HTML visualization of the dataset's knowledge graph.

    This endpoint creates an interactive HTML visualization of the knowledge graph
    for a specific dataset. The visualization displays nodes and edges representing
    entities and their relationships, allowing users to explore the graph structure
    visually.

    ## Query Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to visualize

    ## Response
    Returns an HTML page containing the interactive graph visualization.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist
    - **403 Forbidden**: User doesn't have permission to read the dataset
    - **500 Internal Server Error**: Error generating visualization

    ## Notes
    - User must have read permissions on the dataset
    - Visualization is interactive and allows graph exploration

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        dataset_id=dataset_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    dataset_id: UUID,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Visualize

     Generate an HTML visualization of the dataset's knowledge graph.

    This endpoint creates an interactive HTML visualization of the knowledge graph
    for a specific dataset. The visualization displays nodes and edges representing
    entities and their relationships, allowing users to explore the graph structure
    visually.

    ## Query Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to visualize

    ## Response
    Returns an HTML page containing the interactive graph visualization.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist
    - **403 Forbidden**: User doesn't have permission to read the dataset
    - **500 Internal Server Error**: Error generating visualization

    ## Notes
    - User must have read permissions on the dataset
    - Visualization is interactive and allows graph exploration

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        dataset_id=dataset_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    dataset_id: UUID,
) -> Response[Union[Any, HTTPValidationError]]:
    """Visualize

     Generate an HTML visualization of the dataset's knowledge graph.

    This endpoint creates an interactive HTML visualization of the knowledge graph
    for a specific dataset. The visualization displays nodes and edges representing
    entities and their relationships, allowing users to explore the graph structure
    visually.

    ## Query Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to visualize

    ## Response
    Returns an HTML page containing the interactive graph visualization.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist
    - **403 Forbidden**: User doesn't have permission to read the dataset
    - **500 Internal Server Error**: Error generating visualization

    ## Notes
    - User must have read permissions on the dataset
    - Visualization is interactive and allows graph exploration

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        dataset_id=dataset_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    dataset_id: UUID,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Visualize

     Generate an HTML visualization of the dataset's knowledge graph.

    This endpoint creates an interactive HTML visualization of the knowledge graph
    for a specific dataset. The visualization displays nodes and edges representing
    entities and their relationships, allowing users to explore the graph structure
    visually.

    ## Query Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset to visualize

    ## Response
    Returns an HTML page containing the interactive graph visualization.

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist
    - **403 Forbidden**: User doesn't have permission to read the dataset
    - **500 Internal Server Error**: Error generating visualization

    ## Notes
    - User must have read permissions on the dataset
    - Visualization is interactive and allows graph exploration

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            dataset_id=dataset_id,
        )
    ).parsed
