from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.graph_dto import GraphDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    dataset_id: UUID,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/api/v1/datasets/{dataset_id}/graph",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[GraphDTO, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = GraphDTO.from_dict(response.json())

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
) -> Response[Union[GraphDTO, HTTPValidationError]]:
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
) -> Response[Union[GraphDTO, HTTPValidationError]]:
    """Get Dataset Graph

     Get the knowledge graph visualization for a dataset.

    This endpoint retrieves the knowledge graph data for a specific dataset,
    including nodes and edges that represent the relationships between entities
    in the dataset. The graph data is formatted for visualization purposes.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns the graph data containing:
    - **nodes**: List of graph nodes with id, label, and properties
    - **edges**: List of graph edges with source, target, and label

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving graph data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GraphDTO, HTTPValidationError]]
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
) -> Optional[Union[GraphDTO, HTTPValidationError]]:
    """Get Dataset Graph

     Get the knowledge graph visualization for a dataset.

    This endpoint retrieves the knowledge graph data for a specific dataset,
    including nodes and edges that represent the relationships between entities
    in the dataset. The graph data is formatted for visualization purposes.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns the graph data containing:
    - **nodes**: List of graph nodes with id, label, and properties
    - **edges**: List of graph edges with source, target, and label

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving graph data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GraphDTO, HTTPValidationError]
    """

    return sync_detailed(
        dataset_id=dataset_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    dataset_id: UUID,
    *,
    client: AuthenticatedClient,
) -> Response[Union[GraphDTO, HTTPValidationError]]:
    """Get Dataset Graph

     Get the knowledge graph visualization for a dataset.

    This endpoint retrieves the knowledge graph data for a specific dataset,
    including nodes and edges that represent the relationships between entities
    in the dataset. The graph data is formatted for visualization purposes.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns the graph data containing:
    - **nodes**: List of graph nodes with id, label, and properties
    - **edges**: List of graph edges with source, target, and label

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving graph data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GraphDTO, HTTPValidationError]]
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
) -> Optional[Union[GraphDTO, HTTPValidationError]]:
    """Get Dataset Graph

     Get the knowledge graph visualization for a dataset.

    This endpoint retrieves the knowledge graph data for a specific dataset,
    including nodes and edges that represent the relationships between entities
    in the dataset. The graph data is formatted for visualization purposes.

    ## Path Parameters
    - **dataset_id** (UUID): The unique identifier of the dataset

    ## Response
    Returns the graph data containing:
    - **nodes**: List of graph nodes with id, label, and properties
    - **edges**: List of graph edges with source, target, and label

    ## Error Codes
    - **404 Not Found**: Dataset doesn't exist or user doesn't have access
    - **500 Internal Server Error**: Error retrieving graph data

    Args:
        dataset_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GraphDTO, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            dataset_id=dataset_id,
            client=client,
        )
    ).parsed
