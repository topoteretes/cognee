from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee import datasets
from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.datasets.dto import DataDTO, DeleteDataReceiptDTO
from cognee.api.v1.exceptions import DataNotFoundError
from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.content_hash import normalize_content_hash
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    get_datasets_by_name,
    get_datasets_graph_counts,
)
from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
from cognee.modules.graph.methods import get_formatted_graph_data
from cognee.modules.graph.rdf.sparql import (
    DEFAULT_LIMIT as SPARQL_DEFAULT_LIMIT,
)
from cognee.modules.graph.rdf.sparql import (
    MAX_LIMIT as SPARQL_MAX_LIMIT,
)
from cognee.modules.graph.rdf.sparql import (
    SparqlGraphTooLargeError,
    SparqlQueryError,
    SparqlQueryTimeoutError,
    query_dataset_sparql,
)
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger()


class ErrorResponseDTO(BaseModel):
    message: str


class DataCountDTO(OutDTO):
    count: int


# Shared by GET /status and GET /status/progress — both accept the same
# dataset/pipeline selection, so the query param definitions (alias,
# description, examples) live here once instead of twice.
StatusDatasetIdsQuery = Annotated[
    list[UUID],
    Query(
        alias="dataset",
        description=(
            "Dataset UUIDs to check (from GET /api/v1/datasets)."
            " Omit to get status for all datasets you can read."
        ),
        examples=[["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"]],
    ),
]

StatusPipelineNamesQuery = Annotated[
    list[str],
    Query(
        alias="pipeline",
        description=(
            "Pipeline names to check: 'add_pipeline', 'indexing_pipeline',"
            " 'cognify_pipeline' (the two runs cognify makes, in that order), or"
            " 'code_graph_pipeline' (code ingestion via remember content_type='code')."
            " Omit to default to cognify_pipeline. For the folded memory state see"
            " GET /v1/datasets/status/memory."
        ),
        examples=[["cognify_pipeline"]],
    ),
]


class MemoryStatusDTO(BaseModel):
    """A dataset's memory state, folded from its indexing and cognify runs."""

    state: str = Field(
        examples=["extracting"],
        description=(
            "empty, staged, indexing, indexed, indexing_failed, extracting, ready or "
            "extraction_failed"
        ),
    )
    search_level: str = Field(
        examples=["chunks"],
        description="The best search the dataset answers: none, chunks or graph",
    )
    searchable: bool
    graph_ready: bool
    message: str
    indexing_status: PipelineRunStatus | None = None
    cognify_status: PipelineRunStatus | None = None
    error_class: str | None = None
    error_message: str | None = None


class PipelineRunStatusWithProgress(BaseModel):
    status: PipelineRunStatus
    # Present only once a run has emitted at least one progress tick (see
    # log_pipeline_run_progress); None before that or for terminal runs that
    # predate this field.
    progress: dict[str, Any] | None = Field(
        default=None,
        examples=[{"completed_items": 3, "total_items": 10, "current_stage": "extract_graph"}],
    )


class DataItemProcessingStatusDTO(OutDTO):
    """One data item's completion state for the requested pipeline."""

    id: UUID
    name: str
    completed: bool


class DatasetProcessingStatusDTO(OutDTO):
    """Item-level completion counts for one dataset and one pipeline."""

    total: int = Field(description="Number of data items in the dataset")
    completed: int = Field(description="Items carrying the per-item completion stamp")
    pending: int = Field(description="Items without the stamp (total - completed)")
    items: list[DataItemProcessingStatusDTO] = Field(
        description="One entry per data item, in the same order as GET /datasets/{id}/data"
    )


class DatasetDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime | None = None
    owner_id: UUID


class DatasetGraphSummaryDTO(OutDTO):
    dataset_id: UUID
    pipeline_run_id: UUID | None = None
    num_nodes: int
    num_edges: int
    # None while pipeline_run_id is set means the last count attempt degraded
    # (graph store unavailable) and wasn't cached. The degraded result is held
    # in process briefly (GRAPH_COUNTS_FAILED_TTL_SECONDS, default 15s), so a
    # poll inside that window repeats it rather than recounting.
    computed_at: datetime | None = None


class GraphNodeDTO(OutDTO):
    id: UUID
    label: str
    type: str
    properties: dict


class GraphEdgeDTO(OutDTO):
    source: UUID
    target: UUID
    label: str


class GraphDTO(OutDTO):
    nodes: list[GraphNodeDTO]
    edges: list[GraphEdgeDTO]


class DatasetCreationPayload(InDTO):
    name: str = Field(
        examples=["main_dataset"],
        description=(
            "Name of the dataset to create. If a dataset with this name already exists"
            " for the user, the existing dataset is returned instead of creating a duplicate."
        ),
    )


class DatasetSchemaPayloadDTO(InDTO):
    graph_schema: dict[str, Any] | None = None
    custom_prompt: str | None = None


class SparqlQueryPayloadDTO(InDTO):
    query: str = Field(
        min_length=1,
        description="A read-only SPARQL 1.1 query: SELECT, ASK, CONSTRUCT or DESCRIBE.",
        examples=["SELECT ?s ?name WHERE { ?s rdfs:label ?name } LIMIT 10"],
    )
    base_iri: str | None = Field(
        default=None,
        description=(
            "Prefix for the IRIs minted for nodes without an ontology_uri and for"
            " cognee-native predicates. Defaults to https://cognee.ai/graph/."
        ),
        examples=["https://example.org/memory/"],
    )
    limit: int = Field(
        default=SPARQL_DEFAULT_LIMIT,
        ge=1,
        le=SPARQL_MAX_LIMIT,
        description="Maximum number of SELECT rows to return; sets `truncated` when hit.",
    )


class SparqlResultDTO(BaseModel):
    """One query form's result; the fields for the other forms are omitted.

    Keys are deliberately snake_case (not the API's usual camelCase): they mirror
    the SPARQL JSON results vocabulary callers already parse.
    """

    variables: list[str] | None = Field(default=None, description="SELECT: projected names")
    rows: list[list[str | None]] | None = Field(
        default=None,
        description=(
            "SELECT: one list per solution, aligned with `variables`. IRIs and literal"
            " lexical forms as strings, blank nodes as `_:id`, unbound as null."
        ),
    )
    row_count: int | None = Field(default=None, description="SELECT: len(rows)")
    truncated: bool | None = Field(
        default=None, description="SELECT: true when more rows existed than `limit`"
    )
    boolean: bool | None = Field(default=None, description="ASK: the answer")
    turtle: str | None = Field(
        default=None, description="CONSTRUCT / DESCRIBE: the result graph as Turtle"
    )


def get_datasets_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=list[DatasetDTO])
    async def get_datasets(user: User = Depends(get_authenticated_user)):
        """
        Get all datasets accessible to the authenticated user.

        This endpoint retrieves all datasets that the authenticated user has
        read permissions for. The datasets are returned with their metadata
        including ID, name, creation time, and owner information.

        ## Response
        Returns a list of dataset objects containing:
        - **id**: Unique dataset identifier
        - **name**: Dataset name
        - **created_at**: When the dataset was created
        - **updated_at**: When the dataset was last updated
        - **owner_id**: ID of the dataset owner

        ## Error Codes
        - **500 Internal Server Error**: Error retrieving datasets
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets",
                "cognee_version": cognee_version,
            },
        )

        try:
            datasets = await get_all_user_permission_datasets(user, "read")

            return datasets
        except Exception as error:
            # The exception text can carry database hosts, names or SQL; it
            # belongs in the server log, never in a response.
            logger.exception("Error retrieving datasets")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving datasets.",
            ) from error

    @router.post("", response_model=DatasetDTO)
    async def create_new_dataset(
        dataset_data: DatasetCreationPayload,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Create a new dataset or return existing dataset with the same name.

        This endpoint creates a new dataset with the specified name. If a dataset
        with the same name already exists for the user, it returns the existing
        dataset instead of creating a duplicate. The user is automatically granted
        all permissions (read, write, share, delete) on the created dataset.

        ## Request Parameters
        - **dataset_data** (DatasetCreationPayload): Dataset creation parameters containing:
          - **name**: The name for the new dataset

        ## Response
        Returns the created or existing dataset object containing:
        - **id**: Unique dataset identifier
        - **name**: Dataset name
        - **created_at**: When the dataset was created
        - **updated_at**: When the dataset was last updated
        - **owner_id**: ID of the dataset owner

        ## Error Codes
        - **500 Internal Server Error**: Error creating dataset
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/datasets",
                "cognee_version": cognee_version,
            },
        )

        try:
            datasets = await get_datasets_by_name([dataset_data.name], user.id)

            if datasets:
                return datasets[0]

            dataset = await create_authorized_dataset(dataset_data.name, user)

            return dataset
        except Exception as error:
            # Same as get_datasets: diagnostics stay server-side.
            logger.exception("Error creating dataset")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating dataset.",
            ) from error

    @router.delete("")
    async def delete_all(user: User = Depends(get_authenticated_user)):
        """
        Delete all user's data.

        This endpoint permanently deletes all datasets that user created and all its associated data.
        The user must have delete permissions on the dataset to perform this operation.

        ## Response
        No content returned on successful deletion.
        If no datasets exist for the users, nothing happens.
        """
        await datasets.delete_all(user)

    @router.delete(
        "/{dataset_id}", response_model=None, responses={404: {"model": ErrorResponseDTO}}
    )
    async def delete_dataset(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Delete a dataset by its ID.

        This endpoint permanently deletes a dataset and all its associated data.
        The user must have delete permissions on the dataset to perform this operation.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset to delete

        ## Response
        No content returned on successful deletion.

        ## Error Codes
        - **401/403 Unauthorized/Forbidden**: Dataset doesn't exist or user lacks delete permission
        - **500 Internal Server Error**: Error during deletion
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"DELETE /v1/datasets/{dataset_id!s}",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        await datasets.empty_dataset(dataset_id, user)

    @router.delete(
        "/{dataset_id}/data/{data_id}",
        response_model=DeleteDataReceiptDTO,
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def delete_data(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        data_id: UUID = PathParam(
            description="Data item UUID, from GET /api/v1/datasets/{dataset_id}/data",
            examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Delete a specific data item from a dataset.

        This endpoint removes a specific data item from a dataset while keeping
        the dataset itself intact. The user must have delete permissions on the
        dataset to perform this operation.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset containing the data
        - **data_id** (UUID): The unique identifier of the data item to delete

        ## Response
        A deletion receipt (``DeleteDataReceiptDTO``, camelCase on the wire): ``status``
        (``"success"``, or ``"not_found"`` when neither a Data row nor a graph element
        existed for the id — nothing removed, counts ``0``, still a 200 because deleting
        is idempotent), ``datasetId``, ``dataId`` (the resolved id), ``dataRecordFound``
        (a Data row for the id existed), ``deletedNodes`` / ``deletedEdges`` (graph
        elements removed), ``dataRemaining`` (``false`` when an existence check after the
        delete no longer finds the Data row — the verified outcome; ``null`` when no Data
        row existed) and ``datasetDeleted`` (always ``false`` here).

        Changed (SDK-1150): releases up to 1.6.x answered ``null`` (no body) on success.
        Clients that asserted an empty 200 now receive the receipt; the ``status`` key
        keeps its historical value.

        ## Error Codes
        - **401 Unauthorized**: Dataset doesn't exist or user lacks delete permission
        - **500 Internal Server Error**: The Data row was still present after the delete
          (``DataDeletionIncompleteError``; retry), or another error during deletion

        ## Notes
        A ``data_id`` with no Data row but with graph elements it owns is a custom-graph-
        model deletion: those elements are removed and the receipt says
        ``dataRecordFound: false``, ``dataRemaining: null``. A ``data_id`` that owns
        nothing (mistyped, another dataset's, or already deleted) answers
        ``status: "not_found"`` with zero counts — check ``status``, not just the HTTP
        code, before treating a delete as verified.
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"DELETE /v1/datasets/{dataset_id!s}/data/{data_id!s}",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        return await datasets.delete_data(dataset_id, data_id, user)

    @router.get("/{dataset_id}/graph", response_model=GraphDTO)
    async def get_dataset_graph(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the knowledge graph visualization for a dataset.

        This endpoint retrieves the knowledge graph data for a specific dataset,
        including nodes and edges that represent the relationships between entities
        in the dataset. The graph data is formatted for visualization purposes.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Response
        Returns the graph data containing:
        - **nodes**: List of graph nodes with id, label, type, and properties
        - **edges**: List of graph edges with source, target, and label

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error retrieving graph data
        """

        graph_data = await get_formatted_graph_data(dataset_id, user)

        return graph_data

    @router.post(
        "/{dataset_id}/sparql",
        response_model=SparqlResultDTO,
        response_model_exclude_none=True,
        responses={
            400: {"model": ErrorResponseDTO},
            404: {"model": ErrorResponseDTO},
            413: {"model": ErrorResponseDTO},
            504: {"model": ErrorResponseDTO},
        },
    )
    async def query_dataset_with_sparql(
        payload: SparqlQueryPayloadDTO,
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Run a read-only SPARQL query over a dataset's knowledge graph.

        The dataset graph is read in full and loaded into an in-memory RDF store
        for the duration of the request (the same view `cognee.modules.graph.rdf`
        exports), then the query runs there. Nodes carrying an `ontology_uri` keep
        that IRI; other nodes get `{base_iri}node/{id}` and cognee-native edges
        `{base_iri}prop/{relationship_name}`; `name` becomes `rdfs:label` and
        `is_a` becomes `rdf:type` / `rdfs:subClassOf`.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Request Body
        - **query** (str): SELECT, ASK, CONSTRUCT or DESCRIBE. SPARQL Update
          (INSERT, DELETE, LOAD, CLEAR, DROP, CREATE, ADD, MOVE, COPY) is refused, and
          so are `SERVICE`, `FROM` and `FROM NAMED`: the query reads this dataset's
          graph only.
        - **base_iri** (str, optional): prefix for minted IRIs
        - **limit** (int, 1-10000, default 1000): cap on SELECT rows

        ## Response
        Depends on the query form:
        - SELECT: `variables`, `rows`, `row_count`, `truncated`
        - ASK: `boolean`
        - CONSTRUCT / DESCRIBE: `turtle`

        ## Limits
        - Graphs with more than `SPARQL_MAX_GRAPH_NODES` nodes (default 200000)
          are refused with 413; export the graph as RDF instead.
        - Queries that run longer than `SPARQL_QUERY_TIMEOUT_SECONDS` (default 30)
          are abandoned with 504. The server stops waiting; the worker thread
          finishes on its own.

        ## Error Codes
        - **400 Bad Request**: SPARQL Update keywords, `SERVICE` / `FROM` / `FROM NAMED`,
          or the query does not parse
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **413 Content Too Large**: Dataset graph exceeds the node ceiling
        - **504 Gateway Timeout**: Query exceeded the wall-clock budget
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"POST /v1/datasets/{dataset_id!s}/sparql",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        datasets_found = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not datasets_found:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(
                    message=f"Dataset ({dataset_id!s}) not found."
                ).model_dump(),
            )

        dataset = datasets_found[0]

        try:
            return await query_dataset_sparql(
                dataset.id,
                dataset.owner_id,
                payload.query,
                base_iri=payload.base_iri,
                limit=payload.limit,
            )
        except SparqlGraphTooLargeError as error:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content=ErrorResponseDTO(message=str(error)).model_dump(),
            )
        except SparqlQueryTimeoutError as error:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content=ErrorResponseDTO(message=str(error)).model_dump(),
            )
        except SparqlQueryError as error:
            # Update keywords, external access or a query rdflib cannot parse: the client's input.
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponseDTO(message=str(error)).model_dump(),
            )

    @router.get(
        "/{dataset_id}/data",
        response_model=list[DataDTO],
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def get_dataset_data(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        limit: int = Query(
            100,
            ge=1,
            le=1000,
            description="Maximum number of data items to return.",
        ),
        offset: int = Query(
            0,
            ge=0,
            le=1_000_000,
            description="Number of data items to skip before returning results.",
        ),
        content_hash: str | None = Query(
            None,
            pattern=r"^[0-9a-fA-F]{32}$",
            description=(
                "Return only your data items whose content_hash equals this MD5 hex digest "
                "(the hash of the text/bytes that were added; case-insensitive). "
                "Scoped to rows you added; limit/offset page the matches."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get a page of data items in a dataset.

        This endpoint retrieves data items (documents, files, etc.) that belong
        to a specific dataset. Each data item includes metadata such as name, type,
        creation time, and storage location.

        Results are paginated. The response is capped at **limit** items (100 by
        default), so this endpoint stays fast on datasets with many documents; page
        through with **offset**. Use `GET /{dataset_id}/data/count` for the total.
        That total is uncapped; offsets above 1,000,000 are rejected, so datasets
        larger than 1,001,000 items cannot be fully traversed by this endpoint.
        Concurrent inserts/deletes can shift offsets; this is not a snapshot.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Query Parameters
        - **limit** (int, 1-1000, default 100): Maximum number of items to return
        - **offset** (int, default 0): Number of items to skip
        - **content_hash** (str, optional): Return only the items *you* added holding
          this content (MD5 hex digest of the added text/bytes, as recorded in each
          item's `content_hash`; case-insensitive, anything but 32 hex characters is
          a 422). Scoped to your own rows like ingestion dedup, so another user's
          copy in a shared dataset is not returned. `limit`/`offset` page the matches
          exactly as they page the listing; an empty first page means you added no
          such content.

        ## Response
        Returns a list of data objects containing:
        - **id**: Unique data item identifier
        - **name**: Data item name
        - **created_at**: When the data was added
        - **updated_at**: When the data was last updated
        - **extension**: File extension
        - **mime_type**: MIME type of the data
        - **raw_data_location**: Storage location of the raw data
        - **dataset_id**: ID of the containing dataset
        - **label**: Label attached to the data item at upload, if any
        - **external_metadata**: Stored metadata dict (upload-provided keys merged over
          loader-derived ones), if any
        - **content_hash**: MD5 hex digest of the ingested payload

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error retrieving data
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{dataset_id!s}/data",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import (
            get_dataset_data,
            get_dataset_data_by_content_hash,
        )

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not dataset:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(
                    message=f"Dataset ({dataset_id!s}) not found."
                ).model_dump(),
            )

        dataset_id = dataset[0].id

        if content_hash is not None:
            dataset_data = await get_dataset_data_by_content_hash(
                dataset_id,
                normalize_content_hash(content_hash),
                user,
                limit=limit,
                offset=offset,
            )
        else:
            dataset_data = await get_dataset_data(
                dataset_id=dataset_id, limit=limit, offset=offset, order_by="created_at"
            )

        if dataset_data is None:
            return []

        # Build each DataDTO explicitly rather than jsonable_encoder(data) into
        # a dict. The encoder walks the whole ORM row -- including the large
        # JSON columns DataDTO does not expose (node_set, pipeline_status,
        # system_metadata) -- and response_model=list[DataDTO] then re-validated
        # and re-encoded the result, so every row was serialized three times and
        # most of that work was thrown away.
        #
        # Naming the fields also keeps the dataset_id override safe: the row's
        # own column is nullable, so the requested id has to win, and validating
        # the row first would reject a NULL before we could substitute it.
        return [
            DataDTO(
                id=data.id,
                name=data.name,
                created_at=data.created_at,
                updated_at=data.updated_at,
                extension=data.extension,
                mime_type=data.mime_type,
                raw_data_location=data.raw_data_location,
                dataset_id=dataset_id,
                label=data.label,
                external_metadata=data.external_metadata,
                data_size=data.data_size,
                content_hash=data.content_hash,
            )
            for data in dataset_data
        ]

    @router.get(
        "/{dataset_id}/data/count",
        response_model=DataCountDTO,
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def get_dataset_data_count(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Count the data items in a dataset.

        Exists so callers that only need "how many documents" do not have to
        fetch the items to count them. `GET /{dataset_id}/data` is paginated,
        so its length is a page size, not a total.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Response
        - **count**: Number of data items in the dataset

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error counting data
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{dataset_id!s}/data/count",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import count_dataset_data

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not dataset:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(
                    message=f"Dataset ({dataset_id!s}) not found."
                ).model_dump(),
            )

        return DataCountDTO(count=await count_dataset_data(dataset_id=dataset[0].id))

    @router.get(
        "/status",
        response_model=dict[str, PipelineRunStatus] | dict[str, dict[str, PipelineRunStatus]],
    )
    async def get_dataset_status(
        datasets: StatusDatasetIdsQuery = [],
        pipelines: StatusPipelineNamesQuery = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the processing status of datasets.

        This endpoint retrieves the current processing status of one or more datasets,
        indicating whether they are being processed, have completed processing, or
        encountered errors during pipeline execution.

        ## Query Parameters
        - **dataset** (List[UUID]): List of dataset UUIDs to check status for.
          If omitted, returns status for all datasets the user has read permission on
        - **pipeline** (List[str], optional): One or more pipeline names to check.
          - If omitted, defaults to **cognify_pipeline** (backward-compatible behavior)
          - If one pipeline is provided, response is a flat map
          - If multiple pipelines are provided, response is nested per dataset and pipeline
          - **Available options: add_pipeline, indexing_pipeline, cognify_pipeline,
            code_graph_pipeline**
          - Note: a background code ingest creates its pipeline run only once the
            repository is cloned — a dataset missing from the response means the run
            has not started yet, not that it failed

        ## Response
        Returns status information in one of two shapes:
        - Single pipeline (default): {dataset_id: status}
        - Multiple pipelines: {dataset_id: {pipeline_name: status}}

        Status values:
        - **pending**: Dataset is queued for processing
        - **running**: Dataset is currently being processed
        - **completed**: Dataset processing completed successfully
        - **failed**: Dataset processing encountered an error

        For in-flight progress (files completed / total, current stage), see
        **GET /v1/datasets/status/progress** — a separate endpoint with its own
        fixed response shape, rather than a flag here that would change what
        this endpoint returns depending on how it's called.

        ## Error Codes
        - **403 Forbidden**: The request owner cannot read every requested dataset
        - **409 Conflict**: An unexpected error occurred while retrieving status
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets/status",
                "datasets": [str(dataset_id) for dataset_id in datasets],
                "pipelines": pipelines,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            # Verify user has permission to read dataset
            authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)

            datasets_statuses = await cognee_datasets.get_status(
                [dataset.id for dataset in authorized_datasets],
                pipeline_names=pipelines or None,
            )

            return datasets_statuses
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("Error retrieving dataset statuses")
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset statuses."},
            )

    @router.get(
        "/status/progress",
        response_model=dict[str, PipelineRunStatusWithProgress]
        | dict[str, dict[str, PipelineRunStatusWithProgress]],
    )
    async def get_dataset_progress(
        datasets: StatusDatasetIdsQuery = [],
        pipelines: StatusPipelineNamesQuery = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the processing status of datasets, together with in-flight progress.

        Same dataset/pipeline selection as **GET /v1/datasets/status**, but each
        status value is always an object {status, progress} instead of a bare
        status — a dedicated endpoint rather than a flag on /status, so neither
        endpoint's response shape ever depends on how it was called.

        ## Query Parameters
        - **dataset** (List[UUID]): Dataset UUIDs to check (from GET /api/v1/datasets). Omit to get
          status for all datasets you can read.
        - **pipeline** (List[str]): Pipeline names to check: 'add_pipeline', 'indexing_pipeline',
          'cognify_pipeline', or 'code_graph_pipeline' (code ingestion via remember
          content_type='code'). Omit to default to cognify_pipeline.

        ## Response
        - Single pipeline (default): {dataset_id: {status, progress}}
        - Multiple pipelines: {dataset_id: {pipeline_name: {status, progress}}}

        **progress** is `null` until the first in-flight progress tick, then an
        object with `completed_items`, `total_items`, and `current_stage` —
        present only while the pipeline is running; terminal runs (completed/
        errored) do not carry a progress snapshot.

        ## Error Codes
        - **409 Conflict**: Error retrieving status (e.g. requesting a dataset you don't have
          read permission for)
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets/status/progress",
                "datasets": [str(dataset_id) for dataset_id in datasets],
                "pipelines": pipelines,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            # Verify user has permission to read dataset
            authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)

            datasets_progress = await cognee_datasets.get_progress(
                [dataset.id for dataset in authorized_datasets],
                pipeline_names=pipelines or None,
            )

            return datasets_progress
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("Error retrieving dataset progress")
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset progress."},
            )

    @router.get("/status/memory", response_model=dict[str, MemoryStatusDTO])
    async def get_dataset_memory_status(
        datasets: StatusDatasetIdsQuery = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get the memory state of datasets: what each one can answer right now.

        cognify() (and remember()) build memory in two runs per dataset: indexing
        (chunk search) and cognify (the knowledge graph). This endpoint folds their
        latest runs into one state per dataset, so a client can tell whether a search
        will answer and how good the answers are, instead of reading two pipeline
        statuses.

        ## Query Parameters
        - **dataset** (List[UUID]): Dataset UUIDs to check (from GET /api/v1/datasets). Omit
          to get the state of all datasets you can read.

        ## Response
        `{dataset_id: {state, search_level, searchable, graph_ready, message,
        indexing_status, cognify_status, error_class, error_message}}`. Every requested
        dataset is present; `state` is `empty` when nothing ran on it.

        - **state**: `empty`, `staged`, `indexing`, `indexed`, `indexing_failed`,
          `extracting`, `ready` or `extraction_failed`. During a cognify() build it goes
          `indexing`, `extracting`, then `ready`.
        - **search_level**: `none`, `chunks` (CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION,
          HYBRID_COMPLETION answer) or `graph` (everything answers)
        - **message**: one sentence saying what answers now and what improves next

        ## Error Codes
        - **403 Forbidden**: The request owner cannot read every requested dataset
        - **409 Conflict**: An unexpected error occurred while deriving the state
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets/status/memory",
                "datasets": [str(dataset_id) for dataset_id in datasets],
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.datasets.datasets import datasets as cognee_datasets

        try:
            authorized_datasets = await get_authorized_existing_datasets(datasets, "read", user)
            statuses = await cognee_datasets.get_memory_status(
                [dataset.id for dataset in authorized_datasets]
            )
            return {dataset_id: status.to_dict() for dataset_id, status in statuses.items()}
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("Error deriving dataset memory status")
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to derive dataset memory status."},
            )

    @router.get("/graph-summary", response_model=list[DatasetGraphSummaryDTO])
    async def get_datasets_graph_summary(
        dataset_ids: Annotated[
            list[UUID],
            Query(
                alias="dataset_ids",
                description=(
                    "Dataset UUIDs to summarize (from GET /api/v1/datasets)."
                    " Omit to summarize every dataset you can read."
                ),
                examples=[["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"]],
            ),
        ] = [],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get node/edge counts per dataset, cached per cognify run.

        Counts are computed once per dataset's latest cognify run and cached in
        GraphMetrics, keyed by pipeline_run_id — orders of magnitude cheaper on
        repeat polls than GET /{dataset_id}/graph, which does a full traversal.

        ## Query Parameters
        - **dataset_ids** (List[UUID], optional): Dataset UUIDs to summarize.
          If omitted, summarizes every dataset the user has read permission on.

        ## Response
        Returns a list of summaries containing:
        - **datasetId**: The dataset's UUID
        - **pipelineRunId**: The dataset's latest cognify run, or null if it
          has never been cognified
        - **numNodes** / **numEdges**: Graph size for that run
        - **computedAt**: When the count was cached, or null when it wasn't,
          which happens three ways: the last attempt degraded (graph store
          unavailable, counts are 0 and held for
          GRAPH_COUNTS_FAILED_TTL_SECONDS, default 15s, before another
          attempt); a concurrent caller cached the same run first (counts are
          exact, and the next poll reports a non-null computedAt from that
          row); or caching failed for another reason (counts are exact and
          held for GRAPH_COUNTS_UNCACHED_TTL_SECONDS, default 60s)

        ## Error Codes
        - **409 Conflict**: The summary could not be built (generic message;
          the detail is server-logged rather than returned). A single
          unreadable graph store does not cause this — that dataset comes back
          with zero counts — so this means the relational read itself failed.
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/datasets/graph-summary",
                "dataset_ids": [str(dataset_id) for dataset_id in dataset_ids],
                "cognee_version": cognee_version,
            },
        )

        try:
            authorized_datasets = await get_authorized_existing_datasets(dataset_ids, "read", user)

            if not authorized_datasets:
                return []

            counts = await get_datasets_graph_counts(authorized_datasets)
        except CogneeApiError:
            raise
        except Exception:
            # Same posture as GET /statuses above and the sibling
            # GET /visualize/brains-summary: a poll that fails transiently is a
            # 409 with a generic message, not an unhandled 500 carrying
            # internals to the client. Scoped to the two relational reads
            # this route makes; the DTO construction below is pure Python
            # over an already-validated shape, so a bug there still surfaces
            # as a real 500 instead of being misreported as this endpoint's
            # documented transient-failure case.
            logger.exception("Error retrieving dataset graph summary")
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset graph summary."},
            )

        return [
            DatasetGraphSummaryDTO(
                dataset_id=dataset.id,
                pipeline_run_id=counts[dataset.id].pipeline_run_id,
                num_nodes=counts[dataset.id].num_nodes,
                num_edges=counts[dataset.id].num_edges,
                computed_at=counts[dataset.id].computed_at,
            )
            for dataset in authorized_datasets
        ]

    @router.get("/{dataset_id}/data/{data_id}/raw", response_class=FileResponse)
    async def get_raw_data(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        data_id: UUID = PathParam(
            description="Data item UUID, from GET /api/v1/datasets/{dataset_id}/data",
            examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
        ),
        user: User = Depends(get_authenticated_user),
    ) -> Response:
        """
        Download the raw data file for a specific data item.

        This endpoint allows users to download the original, unprocessed data file
        for a specific data item within a dataset. The file is returned as a direct
        download with appropriate headers.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset containing the data
        - **data_id** (UUID): The unique identifier of the data item to download

        ## Response
        Returns the raw data file as a downloadable response.

        ## Error Codes
        - **404 Not Found**: Data item doesn't exist in the dataset, or its raw file is missing
        - **500 Internal Server Error**: Error accessing the raw data file
        - **501 Not Implemented**: Raw data is stored on an unsupported storage scheme
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{dataset_id!s}/data/{data_id!s}/raw",
                "dataset_id": str(dataset_id),
                "data_id": str(data_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_dataset_data, resolve_data_id

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not dataset:
            return JSONResponse(
                status_code=404, content={"message": f"Dataset ({dataset_id}) not found."}
            )

        # Dataset-scoped lookup: resolves the exact id or, for a row whose
        # identity forked in the dataset-scoping upgrade, its recorded
        # pre-fork legacy_id — every id ever issued keeps resolving.
        resolved_id = await resolve_data_id(dataset[0].id, data_id)

        if resolved_id is None:
            raise DataNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        matching_data = [
            data for data in await get_dataset_data(dataset[0].id) if data.id == resolved_id
        ]

        if len(matching_data) == 0:
            raise DataNotFoundError(
                message=f"Data ({data_id}) not found in dataset ({dataset_id})."
            )

        # Use the data object already verified to belong to the authorized dataset,
        # rather than calling get_data() which checks owner_id and would reject
        # ACL-granted readers who are not the data owner.
        data = matching_data[0]

        raw_location = data.raw_data_location
        parsed_uri = urlparse(raw_location)

        def stream_through_storage(download_name: str) -> StreamingResponse:
            # open_data_file goes through the storage layer, so an S3 object is
            # fetched and a sealed object (FILE_ENCRYPTION_ENABLED) is unsealed;
            # FileResponse would serve the ciphertext bytes as they lie on disk.
            from cognee.infrastructure.files.utils.open_data_file import open_data_file
            from cognee.infrastructure.utils.run_async import run_async

            media_type = data.mime_type or "application/octet-stream"

            async def file_iterator(chunk_size: int = 1024 * 1024):
                async with open_data_file(raw_location, mode="rb") as file:
                    while True:
                        chunk = await run_async(file.read, chunk_size)
                        if not chunk:
                            break
                        yield chunk

            return StreamingResponse(
                file_iterator(),
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
            )

        if parsed_uri.scheme == "s3":
            return stream_through_storage(Path(parsed_uri.path).name or data.name)

        if parsed_uri.scheme in ("file", "") or (
            len(parsed_uri.scheme) == 1 and parsed_uri.scheme.isalpha()
        ):
            from cognee.infrastructure.files.encryption import is_encrypted_local_file
            from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path

            file_path = get_data_file_path(raw_location)
            path = Path(file_path)

            if not path.is_file():
                raise DataNotFoundError(message=f"Raw file not found on disk for data ({data_id}).")

            if is_encrypted_local_file(file_path):
                return stream_through_storage(path.name or data.name)

            return FileResponse(path=path)

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Storage scheme '{parsed_uri.scheme}' not supported for direct download.",
        )

    @router.get("/{dataset_id}/schema", response_model=dict)
    async def get_dataset_schema(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        """Return the stored graph schema and custom prompt for a dataset.

        ## Path Parameters
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        """
        from sqlalchemy import select

        from cognee.modules.data.models import DatasetConfiguration

        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)
        if not dataset:
            return JSONResponse(status_code=404, content={"error": "Dataset not found"})

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            config = await session.scalar(
                select(DatasetConfiguration).where(DatasetConfiguration.dataset_id == dataset_id)
            )
        if not config:
            return {"graph_schema": None, "custom_prompt": None}
        return {
            "graph_schema": config.graph_schema,
            "custom_prompt": config.custom_prompt,
        }

    @router.put("/{dataset_id}/schema", response_model=dict)
    async def update_dataset_schema(
        dataset_id: UUID,
        payload: DatasetSchemaPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        """Store or update the graph schema and custom prompt for a dataset.

        ## Path Parameters
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).

        ## Request Parameters
        - **customPrompt** (Optional[str]): Custom extraction prompt to store for the
          dataset; omitting it leaves any existing prompt unchanged.
        - **graphSchema** (Optional[Dict[str, Any]]): JSON graph schema to store for the
          dataset; omitting it leaves any existing schema unchanged.
        """
        from sqlalchemy import select

        from cognee.modules.data.models import DatasetConfiguration

        dataset = await get_authorized_existing_datasets([dataset_id], "write", user)
        if not dataset:
            return JSONResponse(status_code=404, content={"error": "Dataset not found"})

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            config = await session.scalar(
                select(DatasetConfiguration).where(DatasetConfiguration.dataset_id == dataset_id)
            )
            if config:
                if payload.graph_schema is not None:
                    config.graph_schema = payload.graph_schema
                if payload.custom_prompt is not None:
                    config.custom_prompt = payload.custom_prompt
            else:
                config = DatasetConfiguration(
                    dataset_id=dataset_id,
                    graph_schema=payload.graph_schema,
                    custom_prompt=payload.custom_prompt,
                )
                session.add(config)
            await session.commit()
        return {"status": "ok"}

    @router.get(
        "/{dataset_id}/processing-status",
        response_model=DatasetProcessingStatusDTO,
        responses={404: {"model": ErrorResponseDTO}},
    )
    async def get_dataset_processing_status(
        dataset_id: UUID = PathParam(
            description="Dataset UUID, the id field from GET /api/v1/datasets (not the name)",
            examples=["b8a7c3de-4f5a-4b6c-8d9e-0f1a2b3c4d5e"],
        ),
        pipeline: str = Query(
            "cognify_pipeline",
            description=(
                "Pipeline whose per-item completion to count: 'cognify_pipeline'"
                " (default), 'add_pipeline', 'indexing_pipeline', or 'code_graph_pipeline'."
            ),
            examples=["cognify_pipeline"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get item-level processing status for a dataset.

        `GET /status` reports whether a pipeline *run* is in progress or done for a
        dataset. This endpoint answers the finer question operators need when
        triaging incremental loads: which of the dataset's data items carry the
        per-item completion stamp for a pipeline, and which are still pending.

        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset

        ## Query Parameters
        - **pipeline** (str, optional): Pipeline name to inspect. Defaults to
          `cognify_pipeline`.

        ## Response
        - **total**: Number of data items in the dataset
        - **completed**: Items whose per-item status for the pipeline is completed
          (both the legacy string and the dict status representation are recognised)
        - **pending**: `total - completed`
        - **items**: `[{id, name, completed}]`, one entry per data item, in the same
          order as `GET /datasets/{id}/data`. `id` is the data_id accepted by
          `DELETE /datasets/{id}/data/{data_id}` and `forget(data_id=...)`

        Per-item errored state is not persisted, so it is not reported: a pending
        item may be untouched, in progress, or failed.

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **409 Conflict**: Error computing the status
        """
        send_telemetry(
            "Datasets API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": f"GET /v1/datasets/{dataset_id!s}/processing-status",
                "dataset_id": str(dataset_id),
                "pipeline": pipeline,
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.data.methods import get_dataset_processing_status

        # Verify user has permission to read dataset
        dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

        if not dataset:
            return JSONResponse(
                status_code=404,
                content=ErrorResponseDTO(
                    message=f"Dataset ({dataset_id!s}) not found."
                ).model_dump(),
            )

        try:
            return await get_dataset_processing_status(dataset[0].id, pipeline_name=pipeline)
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("Error retrieving dataset processing status")
            return JSONResponse(
                status_code=409,
                content={"error": "Unable to retrieve dataset processing status."},
            )

    return router
