from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.code_pipeline_index_payload_dto import CodePipelineIndexPayloadDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CodePipelineIndexPayloadDTO,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/code-pipeline/index",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

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
    client: Union[AuthenticatedClient, Client],
    body: CodePipelineIndexPayloadDTO,
) -> Response[Union[Any, HTTPValidationError]]:
    """Code Pipeline Index

     Run indexation on a code repository.

    This endpoint processes a code repository to create a knowledge graph
    of the codebase structure, dependencies, and relationships.

    ## Request Parameters
    - **repo_path** (str): Path to the code repository
    - **include_docs** (bool): Whether to include documentation files (default: false)

    ## Response
    No content returned. Processing results are logged.

    ## Error Codes
    - **409 Conflict**: Error during indexation process

    Args:
        body (CodePipelineIndexPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
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
    body: CodePipelineIndexPayloadDTO,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Code Pipeline Index

     Run indexation on a code repository.

    This endpoint processes a code repository to create a knowledge graph
    of the codebase structure, dependencies, and relationships.

    ## Request Parameters
    - **repo_path** (str): Path to the code repository
    - **include_docs** (bool): Whether to include documentation files (default: false)

    ## Response
    No content returned. Processing results are logged.

    ## Error Codes
    - **409 Conflict**: Error during indexation process

    Args:
        body (CodePipelineIndexPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CodePipelineIndexPayloadDTO,
) -> Response[Union[Any, HTTPValidationError]]:
    """Code Pipeline Index

     Run indexation on a code repository.

    This endpoint processes a code repository to create a knowledge graph
    of the codebase structure, dependencies, and relationships.

    ## Request Parameters
    - **repo_path** (str): Path to the code repository
    - **include_docs** (bool): Whether to include documentation files (default: false)

    ## Response
    No content returned. Processing results are logged.

    ## Error Codes
    - **409 Conflict**: Error during indexation process

    Args:
        body (CodePipelineIndexPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CodePipelineIndexPayloadDTO,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Code Pipeline Index

     Run indexation on a code repository.

    This endpoint processes a code repository to create a knowledge graph
    of the codebase structure, dependencies, and relationships.

    ## Request Parameters
    - **repo_path** (str): Path to the code repository
    - **include_docs** (bool): Whether to include documentation files (default: false)

    ## Response
    No content returned. Processing results are logged.

    ## Error Codes
    - **409 Conflict**: Error during indexation process

    Args:
        body (CodePipelineIndexPayloadDTO):

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
        )
    ).parsed
