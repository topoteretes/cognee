from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.code_pipeline_retrieve_api_v1_code_pipeline_retrieve_post_response_200_item import (
    CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item,
)
from ...models.code_pipeline_retrieve_payload_dto import CodePipelineRetrievePayloadDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CodePipelineRetrievePayloadDTO,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/code-pipeline/retrieve",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, list["CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item"]]]:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

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
) -> Response[Union[HTTPValidationError, list["CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CodePipelineRetrievePayloadDTO,
) -> Response[Union[HTTPValidationError, list["CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item"]]]:
    """Code Pipeline Retrieve

     Retrieve context from the code knowledge graph.

    This endpoint searches the indexed code repository to find relevant
    context based on the provided query.

    ## Request Parameters
    - **query** (str): Search query for code context
    - **full_input** (str): Full input text for processing

    ## Response
    Returns a list of relevant code files and context as JSON.

    ## Error Codes
    - **409 Conflict**: Error during retrieval process

    Args:
        body (CodePipelineRetrievePayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item']]]
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
    body: CodePipelineRetrievePayloadDTO,
) -> Optional[Union[HTTPValidationError, list["CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item"]]]:
    """Code Pipeline Retrieve

     Retrieve context from the code knowledge graph.

    This endpoint searches the indexed code repository to find relevant
    context based on the provided query.

    ## Request Parameters
    - **query** (str): Search query for code context
    - **full_input** (str): Full input text for processing

    ## Response
    Returns a list of relevant code files and context as JSON.

    ## Error Codes
    - **409 Conflict**: Error during retrieval process

    Args:
        body (CodePipelineRetrievePayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item']]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CodePipelineRetrievePayloadDTO,
) -> Response[Union[HTTPValidationError, list["CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item"]]]:
    """Code Pipeline Retrieve

     Retrieve context from the code knowledge graph.

    This endpoint searches the indexed code repository to find relevant
    context based on the provided query.

    ## Request Parameters
    - **query** (str): Search query for code context
    - **full_input** (str): Full input text for processing

    ## Response
    Returns a list of relevant code files and context as JSON.

    ## Error Codes
    - **409 Conflict**: Error during retrieval process

    Args:
        body (CodePipelineRetrievePayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, list['CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item']]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CodePipelineRetrievePayloadDTO,
) -> Optional[Union[HTTPValidationError, list["CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item"]]]:
    """Code Pipeline Retrieve

     Retrieve context from the code knowledge graph.

    This endpoint searches the indexed code repository to find relevant
    context based on the provided query.

    ## Request Parameters
    - **query** (str): Search query for code context
    - **full_input** (str): Full input text for processing

    ## Response
    Returns a list of relevant code files and context as JSON.

    ## Error Codes
    - **409 Conflict**: Error during retrieval process

    Args:
        body (CodePipelineRetrievePayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, list['CodePipelineRetrieveApiV1CodePipelineRetrievePostResponse200Item']]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
