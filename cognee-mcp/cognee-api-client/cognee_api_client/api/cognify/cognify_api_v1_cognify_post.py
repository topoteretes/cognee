from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.cognify_api_v1_cognify_post_response_cognify_api_v1_cognify_post import (
    CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost,
)
from ...models.cognify_payload_dto import CognifyPayloadDTO
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CognifyPayloadDTO,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/api/v1/cognify",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost.from_dict(response.json())

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
) -> Response[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: CognifyPayloadDTO,
) -> Response[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]:
    r"""Cognify

     Transform datasets into structured knowledge graphs through cognitive processing.

    This endpoint is the core of Cognee's intelligence layer, responsible for converting
    raw text, documents, and data added through the add endpoint into semantic knowledge graphs.
    It performs deep analysis to extract entities, relationships, and insights from ingested content.

    ## Processing Pipeline
    1. Document classification and permission validation
    2. Text chunking and semantic segmentation
    3. Entity extraction using LLM-powered analysis
    4. Relationship detection and graph construction
    5. Vector embeddings generation for semantic search
    6. Content summarization and indexing

    ## Request Parameters
    - **datasets** (Optional[List[str]]): List of dataset names to process. Dataset names are resolved
    to datasets owned by the authenticated user.
    - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow
    processing of datasets not owned by the user (if permitted).
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).
    - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If
    provided, this prompt will be used instead of the default prompts for knowledge graph extraction.

    ## Response
    - **Blocking execution**: Complete pipeline run information with entity counts, processing duration,
    and success/failure status
    - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring
    via WebSocket subscription

    ## Error Codes
    - **400 Bad Request**: When neither datasets nor dataset_ids are provided, or when specified
    datasets don't exist
    - **409 Conflict**: When processing fails due to system errors, missing LLM API keys, database
    connection failures, or corrupted content

    ## Example Request
    ```json
    {
        \"datasets\": [\"research_papers\", \"documentation\"],
        \"run_in_background\": false,
        \"custom_prompt\": \"Extract entities focusing on technical concepts and their relationships.
    Identify key technologies, methodologies, and their interconnections.\"
    }
    ```

    ## Notes
    To cognify data in datasets not owned by the user and for which the current user has write
    permission,
    the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

    ## Next Steps
    After successful processing, use the search endpoints to query the generated knowledge graph for
    insights, relationships, and semantic search.

    Args:
        body (CognifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]
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
    body: CognifyPayloadDTO,
) -> Optional[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]:
    r"""Cognify

     Transform datasets into structured knowledge graphs through cognitive processing.

    This endpoint is the core of Cognee's intelligence layer, responsible for converting
    raw text, documents, and data added through the add endpoint into semantic knowledge graphs.
    It performs deep analysis to extract entities, relationships, and insights from ingested content.

    ## Processing Pipeline
    1. Document classification and permission validation
    2. Text chunking and semantic segmentation
    3. Entity extraction using LLM-powered analysis
    4. Relationship detection and graph construction
    5. Vector embeddings generation for semantic search
    6. Content summarization and indexing

    ## Request Parameters
    - **datasets** (Optional[List[str]]): List of dataset names to process. Dataset names are resolved
    to datasets owned by the authenticated user.
    - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow
    processing of datasets not owned by the user (if permitted).
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).
    - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If
    provided, this prompt will be used instead of the default prompts for knowledge graph extraction.

    ## Response
    - **Blocking execution**: Complete pipeline run information with entity counts, processing duration,
    and success/failure status
    - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring
    via WebSocket subscription

    ## Error Codes
    - **400 Bad Request**: When neither datasets nor dataset_ids are provided, or when specified
    datasets don't exist
    - **409 Conflict**: When processing fails due to system errors, missing LLM API keys, database
    connection failures, or corrupted content

    ## Example Request
    ```json
    {
        \"datasets\": [\"research_papers\", \"documentation\"],
        \"run_in_background\": false,
        \"custom_prompt\": \"Extract entities focusing on technical concepts and their relationships.
    Identify key technologies, methodologies, and their interconnections.\"
    }
    ```

    ## Notes
    To cognify data in datasets not owned by the user and for which the current user has write
    permission,
    the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

    ## Next Steps
    After successful processing, use the search endpoints to query the generated knowledge graph for
    insights, relationships, and semantic search.

    Args:
        body (CognifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: CognifyPayloadDTO,
) -> Response[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]:
    r"""Cognify

     Transform datasets into structured knowledge graphs through cognitive processing.

    This endpoint is the core of Cognee's intelligence layer, responsible for converting
    raw text, documents, and data added through the add endpoint into semantic knowledge graphs.
    It performs deep analysis to extract entities, relationships, and insights from ingested content.

    ## Processing Pipeline
    1. Document classification and permission validation
    2. Text chunking and semantic segmentation
    3. Entity extraction using LLM-powered analysis
    4. Relationship detection and graph construction
    5. Vector embeddings generation for semantic search
    6. Content summarization and indexing

    ## Request Parameters
    - **datasets** (Optional[List[str]]): List of dataset names to process. Dataset names are resolved
    to datasets owned by the authenticated user.
    - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow
    processing of datasets not owned by the user (if permitted).
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).
    - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If
    provided, this prompt will be used instead of the default prompts for knowledge graph extraction.

    ## Response
    - **Blocking execution**: Complete pipeline run information with entity counts, processing duration,
    and success/failure status
    - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring
    via WebSocket subscription

    ## Error Codes
    - **400 Bad Request**: When neither datasets nor dataset_ids are provided, or when specified
    datasets don't exist
    - **409 Conflict**: When processing fails due to system errors, missing LLM API keys, database
    connection failures, or corrupted content

    ## Example Request
    ```json
    {
        \"datasets\": [\"research_papers\", \"documentation\"],
        \"run_in_background\": false,
        \"custom_prompt\": \"Extract entities focusing on technical concepts and their relationships.
    Identify key technologies, methodologies, and their interconnections.\"
    }
    ```

    ## Notes
    To cognify data in datasets not owned by the user and for which the current user has write
    permission,
    the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

    ## Next Steps
    After successful processing, use the search endpoints to query the generated knowledge graph for
    insights, relationships, and semantic search.

    Args:
        body (CognifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: CognifyPayloadDTO,
) -> Optional[Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]]:
    r"""Cognify

     Transform datasets into structured knowledge graphs through cognitive processing.

    This endpoint is the core of Cognee's intelligence layer, responsible for converting
    raw text, documents, and data added through the add endpoint into semantic knowledge graphs.
    It performs deep analysis to extract entities, relationships, and insights from ingested content.

    ## Processing Pipeline
    1. Document classification and permission validation
    2. Text chunking and semantic segmentation
    3. Entity extraction using LLM-powered analysis
    4. Relationship detection and graph construction
    5. Vector embeddings generation for semantic search
    6. Content summarization and indexing

    ## Request Parameters
    - **datasets** (Optional[List[str]]): List of dataset names to process. Dataset names are resolved
    to datasets owned by the authenticated user.
    - **dataset_ids** (Optional[List[UUID]]): List of existing dataset UUIDs to process. UUIDs allow
    processing of datasets not owned by the user (if permitted).
    - **run_in_background** (Optional[bool]): Whether to execute processing asynchronously. Defaults to
    False (blocking).
    - **custom_prompt** (Optional[str]): Custom prompt for entity extraction and graph generation. If
    provided, this prompt will be used instead of the default prompts for knowledge graph extraction.

    ## Response
    - **Blocking execution**: Complete pipeline run information with entity counts, processing duration,
    and success/failure status
    - **Background execution**: Pipeline run metadata including pipeline_run_id for status monitoring
    via WebSocket subscription

    ## Error Codes
    - **400 Bad Request**: When neither datasets nor dataset_ids are provided, or when specified
    datasets don't exist
    - **409 Conflict**: When processing fails due to system errors, missing LLM API keys, database
    connection failures, or corrupted content

    ## Example Request
    ```json
    {
        \"datasets\": [\"research_papers\", \"documentation\"],
        \"run_in_background\": false,
        \"custom_prompt\": \"Extract entities focusing on technical concepts and their relationships.
    Identify key technologies, methodologies, and their interconnections.\"
    }
    ```

    ## Notes
    To cognify data in datasets not owned by the user and for which the current user has write
    permission,
    the dataset_id must be used (when ENABLE_BACKEND_ACCESS_CONTROL is set to True).

    ## Next Steps
    After successful processing, use the search endpoints to query the generated knowledge graph for
    insights, relationships, and semantic search.

    Args:
        body (CognifyPayloadDTO):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[CognifyApiV1CognifyPostResponseCognifyApiV1CognifyPost, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
