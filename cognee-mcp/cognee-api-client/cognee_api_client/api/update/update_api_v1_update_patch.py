from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_update_api_v1_update_patch import BodyUpdateApiV1UpdatePatch
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response


def _get_kwargs(
    *,
    body: BodyUpdateApiV1UpdatePatch,
    data_id: UUID,
    dataset_id: UUID,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    json_data_id = str(data_id)
    params["data_id"] = json_data_id

    json_dataset_id = str(dataset_id)
    params["dataset_id"] = json_dataset_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/api/v1/update",
        "params": params,
    }

    _kwargs["files"] = body.to_multipart()

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
    *,
    client: AuthenticatedClient,
    body: BodyUpdateApiV1UpdatePatch,
    data_id: UUID,
    dataset_id: UUID,
) -> Response[Union[Any, HTTPValidationError]]:
    """Update

     Update data in a dataset.

    This endpoint updates existing documents in a specified dataset by providing the data_id of the
    existing document
    to update and the new document with the changes as the data.
    The document is updated, analyzed, and the changes are integrated into the knowledge graph.

    ## Request Parameters
    - **data_id** (UUID): UUID of the document to update in Cognee memory
    - **data** (List[UploadFile]): List of files to upload.
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

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
        data_id (UUID):
        dataset_id (UUID):
        body (BodyUpdateApiV1UpdatePatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        data_id=data_id,
        dataset_id=dataset_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: BodyUpdateApiV1UpdatePatch,
    data_id: UUID,
    dataset_id: UUID,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Update

     Update data in a dataset.

    This endpoint updates existing documents in a specified dataset by providing the data_id of the
    existing document
    to update and the new document with the changes as the data.
    The document is updated, analyzed, and the changes are integrated into the knowledge graph.

    ## Request Parameters
    - **data_id** (UUID): UUID of the document to update in Cognee memory
    - **data** (List[UploadFile]): List of files to upload.
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

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
        data_id (UUID):
        dataset_id (UUID):
        body (BodyUpdateApiV1UpdatePatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        data_id=data_id,
        dataset_id=dataset_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: BodyUpdateApiV1UpdatePatch,
    data_id: UUID,
    dataset_id: UUID,
) -> Response[Union[Any, HTTPValidationError]]:
    """Update

     Update data in a dataset.

    This endpoint updates existing documents in a specified dataset by providing the data_id of the
    existing document
    to update and the new document with the changes as the data.
    The document is updated, analyzed, and the changes are integrated into the knowledge graph.

    ## Request Parameters
    - **data_id** (UUID): UUID of the document to update in Cognee memory
    - **data** (List[UploadFile]): List of files to upload.
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

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
        data_id (UUID):
        dataset_id (UUID):
        body (BodyUpdateApiV1UpdatePatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        data_id=data_id,
        dataset_id=dataset_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: BodyUpdateApiV1UpdatePatch,
    data_id: UUID,
    dataset_id: UUID,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Update

     Update data in a dataset.

    This endpoint updates existing documents in a specified dataset by providing the data_id of the
    existing document
    to update and the new document with the changes as the data.
    The document is updated, analyzed, and the changes are integrated into the knowledge graph.

    ## Request Parameters
    - **data_id** (UUID): UUID of the document to update in Cognee memory
    - **data** (List[UploadFile]): List of files to upload.
    - **datasetId** (Optional[UUID]): UUID of an already existing dataset
    - **node_set** Optional[list[str]]: List of node identifiers for graph organization and access
    control.
             Used for grouping related data points in the knowledge graph.

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
        data_id (UUID):
        dataset_id (UUID):
        body (BodyUpdateApiV1UpdatePatch):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            data_id=data_id,
            dataset_id=dataset_id,
        )
    ).parsed
