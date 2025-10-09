from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.add_api_v1_add_post_response_add_api_v1_add_post import AddApiV1AddPostResponseAddApiV1AddPost
from ...models.body_add_api_v1_add_post import BodyAddApiV1AddPost
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: BodyAddApiV1AddPost,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/add",
    }

    _kwargs["files"] = body.to_multipart()

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AddApiV1AddPostResponseAddApiV1AddPost.from_dict(response.json())

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
) -> Response[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: BodyAddApiV1AddPost,
) -> Response[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]:
    """Add

     Add data to a dataset for processing and knowledge graph construction.

    This endpoint accepts various types of data (files, URLs, GitHub repositories)
    and adds them to a specified dataset for processing. The data is ingested,
    analyzed, and integrated into the knowledge graph.

    ## Request Parameters
    - **data** (List[UploadFile]): List of files to upload. Can also include:
      - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
      - GitHub repository URLs (will be cloned and processed)
      - Regular file uploads
    - **datasetName** (Optional[str]): Name of the dataset to add data to
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during add operation
    - **403 Forbidden**: User doesn't have permission to add to dataset

    ## Notes
    - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL
    is set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (BodyAddApiV1AddPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]
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
    body: BodyAddApiV1AddPost,
) -> Optional[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]:
    """Add

     Add data to a dataset for processing and knowledge graph construction.

    This endpoint accepts various types of data (files, URLs, GitHub repositories)
    and adds them to a specified dataset for processing. The data is ingested,
    analyzed, and integrated into the knowledge graph.

    ## Request Parameters
    - **data** (List[UploadFile]): List of files to upload. Can also include:
      - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
      - GitHub repository URLs (will be cloned and processed)
      - Regular file uploads
    - **datasetName** (Optional[str]): Name of the dataset to add data to
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during add operation
    - **403 Forbidden**: User doesn't have permission to add to dataset

    ## Notes
    - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL
    is set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (BodyAddApiV1AddPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: BodyAddApiV1AddPost,
) -> Response[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]:
    """Add

     Add data to a dataset for processing and knowledge graph construction.

    This endpoint accepts various types of data (files, URLs, GitHub repositories)
    and adds them to a specified dataset for processing. The data is ingested,
    analyzed, and integrated into the knowledge graph.

    ## Request Parameters
    - **data** (List[UploadFile]): List of files to upload. Can also include:
      - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
      - GitHub repository URLs (will be cloned and processed)
      - Regular file uploads
    - **datasetName** (Optional[str]): Name of the dataset to add data to
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during add operation
    - **403 Forbidden**: User doesn't have permission to add to dataset

    ## Notes
    - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL
    is set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (BodyAddApiV1AddPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: BodyAddApiV1AddPost,
) -> Optional[Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]]:
    """Add

     Add data to a dataset for processing and knowledge graph construction.

    This endpoint accepts various types of data (files, URLs, GitHub repositories)
    and adds them to a specified dataset for processing. The data is ingested,
    analyzed, and integrated into the knowledge graph.

    ## Request Parameters
    - **data** (List[UploadFile]): List of files to upload. Can also include:
      - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
      - GitHub repository URLs (will be cloned and processed)
      - Regular file uploads
    - **datasetName** (Optional[str]): Name of the dataset to add data to
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

    Either datasetName or datasetId must be provided.

    ## Response
    Returns information about the add operation containing:
    - Status of the operation
    - Details about the processed data
    - Any relevant metadata from the ingestion process

    ## Error Codes
    - **400 Bad Request**: Neither datasetId nor datasetName provided
    - **409 Conflict**: Error during add operation
    - **403 Forbidden**: User doesn't have permission to add to dataset

    ## Notes
    - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL
    is set to True)
    - datasetId value can only be the UUID of an already existing dataset

    Args:
        body (BodyAddApiV1AddPost):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AddApiV1AddPostResponseAddApiV1AddPost, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
