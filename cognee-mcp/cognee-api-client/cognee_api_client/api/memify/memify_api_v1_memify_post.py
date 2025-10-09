from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.memify_api_v1_memify_post_response_memify_api_v1_memify_post import (
    MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost,
)
from ...models.memify_payload_dto import MemifyPayloadDTO
from ...types import Response


def _get_kwargs(
    *,
    body: MemifyPayloadDTO,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/memify",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]:
    if response.status_code == 200:
        response_200 = MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: MemifyPayloadDTO,
) -> Response[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]:
    """Memify

     Enrichment pipeline in Cognee, can work with already built graphs. If no data is provided existing
    knowledge graph will be used as data,
    custom data can also be provided instead which can be processed with provided extraction and
    enrichment tasks.

    Provided tasks and data will be arranged to run the Cognee pipeline and execute graph
    enrichment/creation.

    ## Request Parameters
    - **extractionTasks** Optional[List[str]]: List of available Cognee Tasks to execute for graph/data
    extraction.
    - **enrichmentTasks** Optional[List[str]]: List of available Cognee Tasks to handle enrichment of
    provided graph/data from extraction tasks.
    - **data** Optional[List[str]]: The data to ingest. Can be any text data when custom extraction and
    enrichment tasks are used.
          Data provided here will be forwarded to the first extraction task in the pipeline as input.
          If no data is provided the whole graph (or subgraph if node_name/node_type is specified) will
    be forwarded
    - **dataset_name** (Optional[str]): Name of the datasets to memify
    - **dataset_id** (Optional[UUID]): List of UUIDs of an already existing dataset
    - **node_name** (Optional[List[str]]):  Filter graph to specific named entities (for targeted
    search). Used when no data is provided.
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during memify operation
    - **403 Forbidden**: User doesn't have permission to use dataset

    ## Notes
    - To memify datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is
    set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (MemifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]
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
    body: MemifyPayloadDTO,
) -> Optional[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]:
    """Memify

     Enrichment pipeline in Cognee, can work with already built graphs. If no data is provided existing
    knowledge graph will be used as data,
    custom data can also be provided instead which can be processed with provided extraction and
    enrichment tasks.

    Provided tasks and data will be arranged to run the Cognee pipeline and execute graph
    enrichment/creation.

    ## Request Parameters
    - **extractionTasks** Optional[List[str]]: List of available Cognee Tasks to execute for graph/data
    extraction.
    - **enrichmentTasks** Optional[List[str]]: List of available Cognee Tasks to handle enrichment of
    provided graph/data from extraction tasks.
    - **data** Optional[List[str]]: The data to ingest. Can be any text data when custom extraction and
    enrichment tasks are used.
          Data provided here will be forwarded to the first extraction task in the pipeline as input.
          If no data is provided the whole graph (or subgraph if node_name/node_type is specified) will
    be forwarded
    - **dataset_name** (Optional[str]): Name of the datasets to memify
    - **dataset_id** (Optional[UUID]): List of UUIDs of an already existing dataset
    - **node_name** (Optional[List[str]]):  Filter graph to specific named entities (for targeted
    search). Used when no data is provided.
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during memify operation
    - **403 Forbidden**: User doesn't have permission to use dataset

    ## Notes
    - To memify datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is
    set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (MemifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: MemifyPayloadDTO,
) -> Response[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]:
    """Memify

     Enrichment pipeline in Cognee, can work with already built graphs. If no data is provided existing
    knowledge graph will be used as data,
    custom data can also be provided instead which can be processed with provided extraction and
    enrichment tasks.

    Provided tasks and data will be arranged to run the Cognee pipeline and execute graph
    enrichment/creation.

    ## Request Parameters
    - **extractionTasks** Optional[List[str]]: List of available Cognee Tasks to execute for graph/data
    extraction.
    - **enrichmentTasks** Optional[List[str]]: List of available Cognee Tasks to handle enrichment of
    provided graph/data from extraction tasks.
    - **data** Optional[List[str]]: The data to ingest. Can be any text data when custom extraction and
    enrichment tasks are used.
          Data provided here will be forwarded to the first extraction task in the pipeline as input.
          If no data is provided the whole graph (or subgraph if node_name/node_type is specified) will
    be forwarded
    - **dataset_name** (Optional[str]): Name of the datasets to memify
    - **dataset_id** (Optional[UUID]): List of UUIDs of an already existing dataset
    - **node_name** (Optional[List[str]]):  Filter graph to specific named entities (for targeted
    search). Used when no data is provided.
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during memify operation
    - **403 Forbidden**: User doesn't have permission to use dataset

    ## Notes
    - To memify datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is
    set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (MemifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: MemifyPayloadDTO,
) -> Optional[Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]]:
    """Memify

     Enrichment pipeline in Cognee, can work with already built graphs. If no data is provided existing
    knowledge graph will be used as data,
    custom data can also be provided instead which can be processed with provided extraction and
    enrichment tasks.

    Provided tasks and data will be arranged to run the Cognee pipeline and execute graph
    enrichment/creation.

    ## Request Parameters
    - **extractionTasks** Optional[List[str]]: List of available Cognee Tasks to execute for graph/data
    extraction.
    - **enrichmentTasks** Optional[List[str]]: List of available Cognee Tasks to handle enrichment of
    provided graph/data from extraction tasks.
    - **data** Optional[List[str]]: The data to ingest. Can be any text data when custom extraction and
    enrichment tasks are used.
          Data provided here will be forwarded to the first extraction task in the pipeline as input.
          If no data is provided the whole graph (or subgraph if node_name/node_type is specified) will
    be forwarded
    - **dataset_name** (Optional[str]): Name of the datasets to memify
    - **dataset_id** (Optional[UUID]): List of UUIDs of an already existing dataset
    - **node_name** (Optional[List[str]]):  Filter graph to specific named entities (for targeted
    search). Used when no data is provided.
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during memify operation
    - **403 Forbidden**: User doesn't have permission to use dataset

    ## Notes
    - To memify datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is
    set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (MemifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, MemifyApiV1MemifyPostResponseMemifyApiV1MemifyPost]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
