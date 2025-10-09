from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.combined_search_result import CombinedSearchResult
from ...models.http_validation_error import HTTPValidationError
from ...models.search_payload_dto import SearchPayloadDTO
from ...models.search_result import SearchResult
from ...types import Response


def _get_kwargs(
    *,
    body: SearchPayloadDTO,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/search",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, Union["CombinedSearchResult", list["SearchResult"], list[Any]]]]:
    if response.status_code == 200:

        def _parse_response_200(data: object) -> Union["CombinedSearchResult", list["SearchResult"], list[Any]]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                response_200_type_0 = []
                _response_200_type_0 = data
                for response_200_type_0_item_data in _response_200_type_0:
                    response_200_type_0_item = SearchResult.from_dict(response_200_type_0_item_data)

                    response_200_type_0.append(response_200_type_0_item)

                return response_200_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_1 = CombinedSearchResult.from_dict(data)

                return response_200_type_1
            except:  # noqa: E722
                pass
            if not isinstance(data, list):
                raise TypeError()
            response_200_type_2 = cast(list[Any], data)

            return response_200_type_2

        response_200 = _parse_response_200(response.json())

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
) -> Response[Union[HTTPValidationError, Union["CombinedSearchResult", list["SearchResult"], list[Any]]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: SearchPayloadDTO,
) -> Response[Union[HTTPValidationError, Union["CombinedSearchResult", list["SearchResult"], list[Any]]]]:
    """Search

     Search for nodes in the graph database.

    This endpoint performs semantic search across the knowledge graph to find
    relevant nodes based on the provided query. It supports different search
    types and can be scoped to specific datasets.

    ## Request Parameters
    - **search_type** (SearchType): Type of search to perform
    - **datasets** (Optional[List[str]]): List of dataset names to search within
    - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to search within
    - **query** (str): The search query string
    - **system_prompt** Optional[str]: System prompt to be used for Completion type searches in Cognee
    - **node_name** Optional[list[str]]: Filter results to specific node_sets defined in the add
    pipeline (for targeted search).
    - **top_k** (Optional[int]): Maximum number of results to return (default: 10)
    - **only_context** bool: Set to true to only return context Cognee will be sending to LLM in
    Completion type searches. This will be returned instead of LLM calls for completion type searches.

    ## Response
    Returns a list of search results containing relevant nodes from the graph.

    ## Error Codes
    - **409 Conflict**: Error during search operation
    - **403 Forbidden**: User doesn't have permission to search datasets (returns empty list)

    ## Notes
    - Datasets sent by name will only map to datasets owned by the request sender
    - To search datasets not owned by the request sender, dataset UUID is needed
    - If permission is denied, returns empty list instead of error

    Args:
        body (SearchPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['CombinedSearchResult', list['SearchResult'], list[Any]]]]
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
    body: SearchPayloadDTO,
) -> Optional[Union[HTTPValidationError, Union["CombinedSearchResult", list["SearchResult"], list[Any]]]]:
    """Search

     Search for nodes in the graph database.

    This endpoint performs semantic search across the knowledge graph to find
    relevant nodes based on the provided query. It supports different search
    types and can be scoped to specific datasets.

    ## Request Parameters
    - **search_type** (SearchType): Type of search to perform
    - **datasets** (Optional[List[str]]): List of dataset names to search within
    - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to search within
    - **query** (str): The search query string
    - **system_prompt** Optional[str]: System prompt to be used for Completion type searches in Cognee
    - **node_name** Optional[list[str]]: Filter results to specific node_sets defined in the add
    pipeline (for targeted search).
    - **top_k** (Optional[int]): Maximum number of results to return (default: 10)
    - **only_context** bool: Set to true to only return context Cognee will be sending to LLM in
    Completion type searches. This will be returned instead of LLM calls for completion type searches.

    ## Response
    Returns a list of search results containing relevant nodes from the graph.

    ## Error Codes
    - **409 Conflict**: Error during search operation
    - **403 Forbidden**: User doesn't have permission to search datasets (returns empty list)

    ## Notes
    - Datasets sent by name will only map to datasets owned by the request sender
    - To search datasets not owned by the request sender, dataset UUID is needed
    - If permission is denied, returns empty list instead of error

    Args:
        body (SearchPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['CombinedSearchResult', list['SearchResult'], list[Any]]]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: SearchPayloadDTO,
) -> Response[Union[HTTPValidationError, Union["CombinedSearchResult", list["SearchResult"], list[Any]]]]:
    """Search

     Search for nodes in the graph database.

    This endpoint performs semantic search across the knowledge graph to find
    relevant nodes based on the provided query. It supports different search
    types and can be scoped to specific datasets.

    ## Request Parameters
    - **search_type** (SearchType): Type of search to perform
    - **datasets** (Optional[List[str]]): List of dataset names to search within
    - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to search within
    - **query** (str): The search query string
    - **system_prompt** Optional[str]: System prompt to be used for Completion type searches in Cognee
    - **node_name** Optional[list[str]]: Filter results to specific node_sets defined in the add
    pipeline (for targeted search).
    - **top_k** (Optional[int]): Maximum number of results to return (default: 10)
    - **only_context** bool: Set to true to only return context Cognee will be sending to LLM in
    Completion type searches. This will be returned instead of LLM calls for completion type searches.

    ## Response
    Returns a list of search results containing relevant nodes from the graph.

    ## Error Codes
    - **409 Conflict**: Error during search operation
    - **403 Forbidden**: User doesn't have permission to search datasets (returns empty list)

    ## Notes
    - Datasets sent by name will only map to datasets owned by the request sender
    - To search datasets not owned by the request sender, dataset UUID is needed
    - If permission is denied, returns empty list instead of error

    Args:
        body (SearchPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['CombinedSearchResult', list['SearchResult'], list[Any]]]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: SearchPayloadDTO,
) -> Optional[Union[HTTPValidationError, Union["CombinedSearchResult", list["SearchResult"], list[Any]]]]:
    """Search

     Search for nodes in the graph database.

    This endpoint performs semantic search across the knowledge graph to find
    relevant nodes based on the provided query. It supports different search
    types and can be scoped to specific datasets.

    ## Request Parameters
    - **search_type** (SearchType): Type of search to perform
    - **datasets** (Optional[List[str]]): List of dataset names to search within
    - **dataset_ids** (Optional[List[UUID]]): List of dataset UUIDs to search within
    - **query** (str): The search query string
    - **system_prompt** Optional[str]: System prompt to be used for Completion type searches in Cognee
    - **node_name** Optional[list[str]]: Filter results to specific node_sets defined in the add
    pipeline (for targeted search).
    - **top_k** (Optional[int]): Maximum number of results to return (default: 10)
    - **only_context** bool: Set to true to only return context Cognee will be sending to LLM in
    Completion type searches. This will be returned instead of LLM calls for completion type searches.

    ## Response
    Returns a list of search results containing relevant nodes from the graph.

    ## Error Codes
    - **409 Conflict**: Error during search operation
    - **403 Forbidden**: User doesn't have permission to search datasets (returns empty list)

    ## Notes
    - Datasets sent by name will only map to datasets owned by the request sender
    - To search datasets not owned by the request sender, dataset UUID is needed
    - If permission is denied, returns empty list instead of error

    Args:
        body (SearchPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['CombinedSearchResult', list['SearchResult'], list[Any]]]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
