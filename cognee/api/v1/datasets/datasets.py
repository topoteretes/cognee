import asyncio
from uuid import UUID

from cognee.api.v1.datasets.dto import DataDTO
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.data.content_hash import compute_content_hash, normalize_content_hash
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.methods import (
    get_authorized_dataset,
    get_authorized_existing_datasets,
    get_dataset_data,
    get_dataset_data_by_content_hash,
    has_dataset_data,
)
from cognee.modules.data.models import Data
from cognee.modules.graph.methods import (
    delete_data_nodes_and_edges,
    delete_dataset_nodes_and_edges,
    has_data_related_nodes,
    legacy_delete,
    try_delete_data_by_graph_provenance,
)
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
from cognee.modules.ingestion import discover_directory_datasets
from cognee.modules.operations import record_operation
from cognee.modules.pipelines.operations.get_pipeline_status import (
    get_pipeline_progress,
    get_pipeline_status,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def _fan_out_by_pipeline(dataset_ids: list[UUID], pipeline_names: list[str] | None, fetch):
    """Shared flat/nested shaping for get_status and get_progress.

    ``fetch`` is get_pipeline_status or get_pipeline_progress — only the
    per-dataset value type differs (a bare status vs. {status, progress});
    the flat-vs-nested decision based on how many pipeline names were
    requested is identical either way, so it lives here once.
    """
    # Backward-compatible default: cognify-only flat map.
    if not pipeline_names:
        return await fetch(dataset_ids, pipeline_name="cognify_pipeline")

    # Preserve order while removing duplicates.
    requested_pipelines = list(dict.fromkeys(pipeline_names))

    # For one pipeline, keep flat shape.
    if len(requested_pipelines) == 1:
        return await fetch(dataset_ids, pipeline_name=requested_pipelines[0])

    # For multiple pipelines, return nested shape.
    statuses_by_dataset = {str(dataset_id): {} for dataset_id in dataset_ids}
    for pipeline_name in requested_pipelines:
        pipeline_status = await fetch(dataset_ids, pipeline_name=pipeline_name)
        for dataset_id, status in pipeline_status.items():
            statuses_by_dataset.setdefault(dataset_id, {})[pipeline_name] = status

    return statuses_by_dataset


async def _invalidate_sessions_for_dataset_nonfatal(dataset_id: UUID) -> None:
    """Drop sessions attributed to a deleted dataset. Never fails the delete."""
    try:
        from cognee.modules.session_lifecycle.invalidate_sessions import (
            invalidate_sessions_for_dataset,
        )

        await invalidate_sessions_for_dataset(dataset_id)
    except Exception as error:
        logger.warning(
            "Session invalidation after dataset delete failed (non-fatal): %s", error, exc_info=True
        )


async def _invalidate_sessions_for_deleted_data_nonfatal(
    dataset_id: UUID,
    deleted_elements: DeletedGraphElements | None,
    user_id: UUID | None = None,
) -> None:
    """Remove session entries that used the deleted elements. Never fails the delete."""
    if deleted_elements is None:
        return
    try:
        from cognee.modules.session_lifecycle.invalidate_sessions import (
            invalidate_sessions_for_deleted_data,
        )

        await invalidate_sessions_for_deleted_data(
            dataset_id,
            deleted_elements.node_ids,
            deleted_elements.edge_ids,
            user_id=user_id,
        )
    except Exception as error:
        logger.warning(
            "Session invalidation after data delete failed (non-fatal): %s", error, exc_info=True
        )


def _is_reference_string(text: str, *, check_local_files: bool = True) -> bool:
    """True when ``add(text)`` would have ingested a file or page, not the text.

    Mirrors the string branch of ``save_data_item_to_storage``: an ``s3://``,
    ``http(s)://`` or ``file://`` URL is always a reference, and any other
    string is one when it names an existing local file the ingestion
    allowlist accepts. A ``"/"``-prefixed note that is not a file stays text
    on both sides.

    ``check_local_files=False`` skips that last probe. Under ``serve()``,
    ``add()`` resolved the string against the server's filesystem, so a file
    that happens to exist on this machine says nothing about how it was stored.
    """
    from urllib.parse import urlparse

    if urlparse(text).scheme in ("s3", "http", "https", "file"):
        return True

    if not check_local_files:
        return False

    from cognee.tasks.ingestion.save_data_item_to_storage import _resolve_local_file_uri

    return _resolve_local_file_uri(text) is not None


class datasets:
    """
    Dataset management namespace for Cognee.

    All methods are static and provide operations for listing, inspecting,
    and deleting datasets and the data items within them.

    Example:
        ```python
        import cognee

        # List all accessible datasets
        all_datasets = await cognee.datasets.list_datasets()

        # Check cognify processing status for datasets
        status = await cognee.datasets.get_status([dataset_id])

        # Delete a specific data item from a dataset
        await cognee.datasets.delete_data(dataset_id=dataset_id, data_id=data_id)
        ```
    """

    @staticmethod
    async def list_datasets(user: User | None = None):
        if user is None:
            user = await get_default_user()

        return await get_authorized_existing_datasets([], "read", user)

    @staticmethod
    def discover_datasets(directory_path: str):
        return list(discover_directory_datasets(directory_path).keys())

    @staticmethod
    async def list_data(dataset_id: UUID, user: User | None = None):
        # Route to the remote instance when connected via serve(): the dataset
        # lives on the server, so the local store would report it missing.
        # Rows are parsed through the same DTO the server serializes them
        # with, so callers read the same attributes (``row.id`` as a UUID,
        # ``row.mime_type`` not ``mimeType``) in both modes.
        from cognee.api.v1.serve.state import get_remote_client
        from cognee.modules.data.methods import get_dataset_data

        client = get_remote_client()
        if client is not None:
            return [DataDTO.model_validate(row) for row in await client.list_data(dataset_id)]

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await get_dataset_data(dataset.id, order_by="created_at")

    @staticmethod
    async def sparql(
        dataset_id: UUID,
        query: str,
        *,
        base_iri: str | None = None,
        limit: int = 1000,
        user: User | None = None,
    ) -> dict:
        """Run a read-only SPARQL query over one dataset's graph.

        Same contract as ``POST /api/v1/datasets/{dataset_id}/sparql``: the
        dataset graph is materialized into an in-memory RDF store per call and
        the query runs there. Returns ``{"variables", "rows", "row_count",
        "truncated"}`` for SELECT, ``{"boolean"}`` for ASK and ``{"turtle"}``
        for CONSTRUCT/DESCRIBE. Update forms are refused
        (``SparqlUpdateNotAllowedError``), graphs above
        ``SPARQL_MAX_GRAPH_NODES`` raise ``SparqlGraphTooLargeError`` and a
        query over ``SPARQL_QUERY_TIMEOUT_SECONDS`` raises
        ``SparqlQueryTimeoutError`` (all in ``cognee.modules.graph.rdf``).

        Routes to the remote instance when connected via ``serve()``, like
        ``list_data``; there the server's limits apply and a refused query
        surfaces as ``RuntimeError`` carrying the HTTP status.
        """
        from cognee.api.v1.serve.state import get_remote_client
        from cognee.modules.graph.rdf.sparql import query_dataset_sparql

        client = get_remote_client()
        if client is not None:
            return await client.sparql(dataset_id, query, base_iri=base_iri, limit=limit)

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)
        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        return await query_dataset_sparql(
            dataset.id, dataset.owner_id, query, base_iri=base_iri, limit=limit
        )

    @staticmethod
    async def find_data(
        dataset_id: UUID,
        *,
        content: str | bytes | None = None,
        content_hash: str | None = None,
        user: User | None = None,
    ) -> list[Data]:
        """Find the data items *you* added to a dataset with a given content.

        Pass exactly one of ``content`` or ``content_hash``:

        - ``content`` is the **literal** text or bytes that were added, hashed
          locally with ``compute_content_hash``. It is not resolved the way
          ``add()`` resolves a string: a path, ``file://`` URI, ``s3://`` or
          ``http(s)://`` URL is refused with ``ValueError`` rather than hashed
          as text (which would match nothing and read as "not present"). For
          content that was added by reference, pass the ``content_hash`` read
          off the ``Data`` row or the listing instead. A path- or URL-shaped
          string that was stored as literal text (cognee does this for DLT rows
          and session bridging) is looked up with
          ``content_hash=compute_content_hash(text)``. Under ``serve()`` only
          URL schemes are refused: local files say nothing about the server.
        - ``content_hash`` is a hash computed earlier or read off another
          ``Data`` row; case and surrounding whitespace are normalized, and a
          value that is not a 32-character hex digest is a ``ValueError``.

        A row matches on its ``content_hash`` (the ingested payload) only —
        never on the loader-derived ``raw_content_hash``, so a PDF whose
        extracted text equals ``content`` is not returned. Matches are scoped
        to the caller's own rows (owner and tenant), exactly like ingestion
        dedup: in a shared dataset another user's copy of the same content is
        not yours to find or delete. Newest first; an empty list means you
        added no such content to the dataset.

        Routes to the remote instance when connected via ``serve()``, like
        ``list_data``; the client re-checks every returned row's hash and
        raises ``RuntimeError`` against a server that predates the lookup,
        so an old server can never make the whole dataset look like a match.
        Requires read access to the dataset.
        """
        if (content is None) == (content_hash is None):
            raise ValueError("find_data takes exactly one of `content` or `content_hash`.")

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()

        if content_hash is None:
            if isinstance(content, str) and _is_reference_string(
                content, check_local_files=client is None
            ):
                raise ValueError(
                    "find_data(content=...) takes the literal text or bytes that were added; "
                    f"{content!r} is a path or URL that add() would have resolved to a file "
                    "or page. For content added by reference, pass content_hash= from the "
                    "Data row or the listing; if this exact string was stored as text, pass "
                    "content_hash=compute_content_hash(text) "
                    "(from cognee.modules.data.content_hash)."
                )
            content_hash = compute_content_hash(content)
        else:
            content_hash = normalize_content_hash(content_hash)

        if client is not None:
            return [
                DataDTO.model_validate(row)
                for row in await client.list_data(dataset_id, content_hash=content_hash)
            ]

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await get_dataset_data_by_content_hash(dataset.id, content_hash, user)

    @staticmethod
    async def has_data(dataset_id: str, user: User | None = None) -> bool:
        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id)

        return await has_dataset_data(dataset.id)

    @staticmethod
    async def get_status(dataset_ids: list[UUID], pipeline_names: list[str] | None = None) -> dict:
        return await _fan_out_by_pipeline(dataset_ids, pipeline_names, get_pipeline_status)

    @staticmethod
    async def get_progress(
        dataset_ids: list[UUID], pipeline_names: list[str] | None = None
    ) -> dict:
        """Same flat-or-nested shape as get_status, but each value is
        {status, progress} instead of a bare status. A separate method
        rather than a flag on get_status, so get_status's response shape
        never depends on how it was called.
        """
        return await _fan_out_by_pipeline(dataset_ids, pipeline_names, get_pipeline_progress)

    @staticmethod
    async def empty_dataset(dataset_id: UUID, user: User | None = None):
        from cognee.modules.data.methods import delete_data, delete_dataset
        from cognee.modules.search.operations import delete_dataset_search_history

        if not user:
            user = await get_default_user()

        dataset = await get_authorized_dataset(user, dataset_id, "delete")

        if not dataset:
            raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

        # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
        # on this dataset and exclude concurrent deletes.
        async with (
            dataset_lock(dataset.id),
            set_database_global_context_variables(dataset.id, dataset.owner_id),
        ):
            # Every user's search history on the dataset keeps answer text that
            # can quote its chunks. Purged first and not caught: if it fails,
            # nothing is deleted yet and the same delete can be retried.
            await delete_dataset_search_history(dataset.id)

            deleted_elements = await delete_dataset_nodes_and_edges(dataset_id, user.id)

            # Session memory derived from this dataset would keep asserting
            # the deleted content (stale QA replay / session-context leak),
            # so drop the attributed sessions with the dataset.
            await _invalidate_sessions_for_dataset_nonfatal(dataset.id)
            await _invalidate_sessions_for_deleted_data_nonfatal(
                dataset.id, deleted_elements, user.id
            )

            # delete_dataset removes the dataset's scoped Data rows
            # (files refcounted by raw_data_location) with the record.
            result = await delete_dataset(dataset)

        return result

    @staticmethod
    async def delete_data(
        dataset_id: UUID,
        data_id: UUID,
        user: User | None = None,
        mode: str = "soft",  # mode is there for backwards compatibility. Don't use "hard", it is dangerous.
        delete_dataset_if_empty: bool = False,  # if this flag is True, delete the whole dataset if it is left empty after data deletion
    ):
        async with record_operation(
            "delete", user=user, dataset_id=dataset_id
        ) as operation_context:
            from cognee.modules.data.methods import delete_data, delete_dataset, get_data
            from cognee.modules.search.operations import delete_dataset_search_history

            if not user:
                user = await get_default_user()
            operation_context.set_user(user)

            try:
                dataset = await get_authorized_dataset(user, dataset_id, "delete")
            except PermissionDeniedError:
                raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

            if not dataset:
                raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")

            # Same per-dataset lock as pipeline runs: wait for any in-flight pipeline
            # on this dataset and exclude concurrent deletes.
            async with dataset_lock(dataset.id):
                # Every id ever issued keeps resolving: exact id first, then the
                # recorded original of a backfill-split row (legacy_id).
                from cognee.modules.data.methods import resolve_data_id

                resolved_id = await resolve_data_id(dataset.id, data_id)
                if resolved_id is not None:
                    data_id = resolved_id

                dataset_data = [
                    data for data in await get_dataset_data(dataset.id) if data.id == data_id
                ]

                data = dataset_data[0] if len(dataset_data) > 0 else None

                if not data:
                    # If data is not found in the system, user is using a custom graph model.
                    async with set_database_global_context_variables(dataset_id, dataset.owner_id):
                        deleted_elements = await delete_data_nodes_and_edges(
                            dataset_id, data_id, user.id
                        )
                        await _invalidate_sessions_for_deleted_data_nonfatal(
                            dataset.id, deleted_elements, user.id
                        )

                        dataset_data = await get_dataset_data(dataset.id)
                        if not dataset_data and delete_dataset_if_empty:
                            # As in empty_dataset: history first, then the dataset.
                            await delete_dataset_search_history(dataset.id)
                            await delete_dataset(dataset)

                    return {"status": "success"}

                if str(data.dataset_id) != str(dataset_id):
                    raise UnauthorizedDataAccessError(f"Data {data_id} not accessible.")

                async with set_database_global_context_variables(dataset_id, dataset.owner_id):
                    # Delete mode is exclusive: ledger rows imply the relational-ledger
                    # path; only ledger-free data probes the graph marker to distinguish
                    # graph-provenance data from legacy data without graph ownership.
                    if await has_data_related_nodes(dataset_id, data_id):
                        deleted_elements = await delete_data_nodes_and_edges(
                            dataset_id, data_id, user.id
                        )
                    else:
                        provenance_result = await try_delete_data_by_graph_provenance(
                            dataset_id, data_id
                        )
                        if provenance_result is not None:
                            deleted_elements = DeletedGraphElements.from_source_ref_removal(
                                provenance_result
                            )
                        elif hasattr(await get_graph_engine(), "get_document_subgraph"):
                            deleted_elements = await legacy_delete(data, "soft")
                        else:
                            # An unmarked graph on an adapter without the legacy
                            # subgraph API (the Postgres demo) holds no legacy
                            # data: it is a document that was never cognified,
                            # so there is nothing in the graph to delete.
                            deleted_elements = DeletedGraphElements()

                    await _invalidate_sessions_for_deleted_data_nonfatal(
                        dataset.id, deleted_elements, user.id
                    )

                    await delete_data(data, dataset_id)

                    dataset_data = await get_dataset_data(dataset.id)
                    if not dataset_data and delete_dataset_if_empty:
                        # As in empty_dataset: history first, then the dataset.
                        await delete_dataset_search_history(dataset.id)
                        await delete_dataset(dataset)

            return {"status": "success"}

    @staticmethod
    async def delete_all(user: User | None = None):
        if not user:
            user = await get_default_user()

        user_datasets = await get_authorized_existing_datasets([], "delete", user)

        for dataset in user_datasets:
            await datasets.empty_dataset(dataset.id, user)
