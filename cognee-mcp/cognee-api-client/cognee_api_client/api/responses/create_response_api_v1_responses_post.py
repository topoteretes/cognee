from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.response_body import ResponseBody
from ...models.response_request import ResponseRequest
from ...types import Response


def _get_kwargs(
    *,
    body: ResponseRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/responses/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, ResponseBody]]:
    if response.status_code == 200:
        response_200 = ResponseBody.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, ResponseBody]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: ResponseRequest,
) -> Response[Union[HTTPValidationError, ResponseBody]]:
    r"""Create Response

     OpenAI-compatible responses endpoint with function calling support.

    This endpoint provides OpenAI-compatible API responses with integrated
    function calling capabilities for Cognee operations.

    ## Request Parameters
    - **input** (str): The input text to process
    - **model** (str): The model to use for processing
    - **tools** (Optional[List[Dict]]): Available tools for function calling
    - **tool_choice** (Any): Tool selection strategy (default: \"auto\")
    - **temperature** (float): Response randomness (default: 1.0)

    ## Response
    Returns an OpenAI-compatible response body with function call results.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **500 Internal Server Error**: Error processing request

    ## Notes
    - Compatible with OpenAI API format
    - Supports function calling with Cognee tools
    - Uses default tools if none provided

    Args:
        body (ResponseRequest): Request body for the new responses endpoint (OpenAI Responses API
            format)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ResponseBody]]
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
    body: ResponseRequest,
) -> Optional[Union[HTTPValidationError, ResponseBody]]:
    r"""Create Response

     OpenAI-compatible responses endpoint with function calling support.

    This endpoint provides OpenAI-compatible API responses with integrated
    function calling capabilities for Cognee operations.

    ## Request Parameters
    - **input** (str): The input text to process
    - **model** (str): The model to use for processing
    - **tools** (Optional[List[Dict]]): Available tools for function calling
    - **tool_choice** (Any): Tool selection strategy (default: \"auto\")
    - **temperature** (float): Response randomness (default: 1.0)

    ## Response
    Returns an OpenAI-compatible response body with function call results.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **500 Internal Server Error**: Error processing request

    ## Notes
    - Compatible with OpenAI API format
    - Supports function calling with Cognee tools
    - Uses default tools if none provided

    Args:
        body (ResponseRequest): Request body for the new responses endpoint (OpenAI Responses API
            format)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ResponseBody]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: ResponseRequest,
) -> Response[Union[HTTPValidationError, ResponseBody]]:
    r"""Create Response

     OpenAI-compatible responses endpoint with function calling support.

    This endpoint provides OpenAI-compatible API responses with integrated
    function calling capabilities for Cognee operations.

    ## Request Parameters
    - **input** (str): The input text to process
    - **model** (str): The model to use for processing
    - **tools** (Optional[List[Dict]]): Available tools for function calling
    - **tool_choice** (Any): Tool selection strategy (default: \"auto\")
    - **temperature** (float): Response randomness (default: 1.0)

    ## Response
    Returns an OpenAI-compatible response body with function call results.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **500 Internal Server Error**: Error processing request

    ## Notes
    - Compatible with OpenAI API format
    - Supports function calling with Cognee tools
    - Uses default tools if none provided

    Args:
        body (ResponseRequest): Request body for the new responses endpoint (OpenAI Responses API
            format)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ResponseBody]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: ResponseRequest,
) -> Optional[Union[HTTPValidationError, ResponseBody]]:
    r"""Create Response

     OpenAI-compatible responses endpoint with function calling support.

    This endpoint provides OpenAI-compatible API responses with integrated
    function calling capabilities for Cognee operations.

    ## Request Parameters
    - **input** (str): The input text to process
    - **model** (str): The model to use for processing
    - **tools** (Optional[List[Dict]]): Available tools for function calling
    - **tool_choice** (Any): Tool selection strategy (default: \"auto\")
    - **temperature** (float): Response randomness (default: 1.0)

    ## Response
    Returns an OpenAI-compatible response body with function call results.

    ## Error Codes
    - **400 Bad Request**: Invalid request parameters
    - **500 Internal Server Error**: Error processing request

    ## Notes
    - Compatible with OpenAI API format
    - Supports function calling with Cognee tools
    - Uses default tools if none provided

    Args:
        body (ResponseRequest): Request body for the new responses endpoint (OpenAI Responses API
            format)

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ResponseBody]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
