from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_dataset_status_api_v1_datasets_status_get_response_get_dataset_status_api_v1_datasets_status_get import (
    GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet,
)
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    dataset: Union[Unset, list[UUID]] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_dataset: Union[Unset, list[str]] = UNSET
    if not isinstance(dataset, Unset):
        json_dataset = []
        for dataset_item_data in dataset:
            dataset_item = str(dataset_item_data)
            json_dataset.append(dataset_item)

    params["dataset"] = json_dataset

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/api/v1/datasets/status",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[
    Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
]:
    if response.status_code == 200:
        response_200 = GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet.from_dict(
            response.json()
        )

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
) -> Response[
    Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    dataset: Union[Unset, list[UUID]] = UNSET,
) -> Response[
    Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
]:
    """Get Dataset Status

     Get the processing status of datasets.

    This endpoint retrieves the current processing status of one or more datasets,
    indicating whether they are being processed, have completed processing, or
    encountered errors during pipeline execution.

    ## Query Parameters
    - **dataset** (List[UUID]): List of dataset UUIDs to check status for

    ## Response
    Returns a dictionary mapping dataset IDs to their processing status:
    - **pending**: Dataset is queued for processing
    - **running**: Dataset is currently being processed
    - **completed**: Dataset processing completed successfully
    - **failed**: Dataset processing encountered an error

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving status information

    Args:
        dataset (Union[Unset, list[UUID]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        dataset=dataset,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    dataset: Union[Unset, list[UUID]] = UNSET,
) -> Optional[
    Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
]:
    """Get Dataset Status

     Get the processing status of datasets.

    This endpoint retrieves the current processing status of one or more datasets,
    indicating whether they are being processed, have completed processing, or
    encountered errors during pipeline execution.

    ## Query Parameters
    - **dataset** (List[UUID]): List of dataset UUIDs to check status for

    ## Response
    Returns a dictionary mapping dataset IDs to their processing status:
    - **pending**: Dataset is queued for processing
    - **running**: Dataset is currently being processed
    - **completed**: Dataset processing completed successfully
    - **failed**: Dataset processing encountered an error

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving status information

    Args:
        dataset (Union[Unset, list[UUID]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        dataset=dataset,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    dataset: Union[Unset, list[UUID]] = UNSET,
) -> Response[
    Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
]:
    """Get Dataset Status

     Get the processing status of datasets.

    This endpoint retrieves the current processing status of one or more datasets,
    indicating whether they are being processed, have completed processing, or
    encountered errors during pipeline execution.

    ## Query Parameters
    - **dataset** (List[UUID]): List of dataset UUIDs to check status for

    ## Response
    Returns a dictionary mapping dataset IDs to their processing status:
    - **pending**: Dataset is queued for processing
    - **running**: Dataset is currently being processed
    - **completed**: Dataset processing completed successfully
    - **failed**: Dataset processing encountered an error

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving status information

    Args:
        dataset (Union[Unset, list[UUID]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        dataset=dataset,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    dataset: Union[Unset, list[UUID]] = UNSET,
) -> Optional[
    Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
]:
    """Get Dataset Status

     Get the processing status of datasets.

    This endpoint retrieves the current processing status of one or more datasets,
    indicating whether they are being processed, have completed processing, or
    encountered errors during pipeline execution.

    ## Query Parameters
    - **dataset** (List[UUID]): List of dataset UUIDs to check status for

    ## Response
    Returns a dictionary mapping dataset IDs to their processing status:
    - **pending**: Dataset is queued for processing
    - **running**: Dataset is currently being processed
    - **completed**: Dataset processing completed successfully
    - **failed**: Dataset processing encountered an error

    ## Error Codes
    - **500 Internal Server Error**: Error retrieving status information

    Args:
        dataset (Union[Unset, list[UUID]]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[GetDatasetStatusApiV1DatasetsStatusGetResponseGetDatasetStatusApiV1DatasetsStatusGet, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            dataset=dataset,
        )
    ).parsed
