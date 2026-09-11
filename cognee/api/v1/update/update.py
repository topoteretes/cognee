from time import perf_counter
from typing import Any, BinaryIO
from uuid import UUID

from pydantic import BaseModel

from cognee.api.v1.add import add
from cognee.api.v1.cognify import cognify
from cognee.api.v1.forget import forget
from cognee.api.v1.update.incremental import (
    IncrementalUpdateNotPossible,
    RefusalReason,
    incremental_update,
    recorded_chunk_budget,
)
from cognee.api.v1.update.result import Fallback, UpdateBatchResult, UpdateError, UpdateResult
from cognee.modules.chunking.chunk_policy import DEFAULT_CHUNK_POLICY, ChunkPolicy
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger

logger = get_logger("update")


async def update(
    data: BinaryIO | list[BinaryIO] | str | list[str] | Any,
    dataset_id: UUID,
    data_id: UUID | None = None,
    user: User = None,
    node_set: list[str] | None = None,
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    preferred_loaders: dict[str, dict[str, Any]] | None = None,
    incremental_loading: bool = True,
    data_cache: bool = True,
    chunk_level_diff: bool = True,
    graph_model: type[BaseModel] = KnowledgeGraph,
    custom_prompt: str | None = None,
    chunker: type = TextChunker,
    policy: ChunkPolicy = DEFAULT_CHUNK_POLICY,
) -> dict:
    """
    Update existing data in Cognee.

    The document keeps its ``data_id`` across updates — on EVERY path. The
    target is resolved first (from ``data_id``, exact or the recorded pre-fork
    ``legacy_id``, or inferred from the input's origin), the chunk-level
    incremental path operates on the resolved row in place, and the full
    rebuild drops the document's memory and refreshes the same row with a
    pinned re-add — the row is never deleted, so externally held id mappings
    never break and a failed rebuild leaves a document to re-cognify rather
    than a document that is gone. update() never creates documents; use add()
    for that.

    Which document: ``data_id`` names it. Without one, update() infers it from
    the input's origin — a local file by its path, an upload by its filename —
    against the dataset's documents. Raw text (stored
    under a content-hash name), a renamed or moved file, an origin no document
    came from, or one that several documents share cannot be matched and raise
    ``UpdateTargetNotInferredError`` (422) before anything is written; the
    message says how to list the dataset's documents and pass ``data_id``. An
    id that resolves to no document raises ``UpdateTargetNotFoundError`` (404).

    Batches: a list of several inputs, or a directory, updates one document
    per file and returns ``UpdateBatchResult`` — the per-document results plus
    the counts for "N updated, N unchanged, N failed" (a one-item list is that
    one document, as before). Every target is resolved
    before the first document is touched, so an input that cannot be matched
    fails the whole call with nothing written; a document whose update raises
    after that is recorded as ``failed`` with the error and the batch carries
    on, so the caller retries it by ``data_id``. A single ``data_id`` goes
    with a single input; per-document ids travel as ``DataItem(data=...,
    data_id=...)``.

    Supported Input Types:
        - **Text strings**: Direct text content (str) - any string not starting with "/" or "file://"
        - **File paths**: Local file paths as strings in these formats:
            * Absolute paths: "/path/to/document.pdf"
            * File URLs: "file:///path/to/document.pdf" or "file://relative/path.txt"
            * S3 paths: "s3://bucket-name/path/to/file.pdf"
        - **Directories**: every file under the directory, as a batch
        - **Binary file objects**: File handles/streams (BinaryIO)

    Supported File Formats:
        - Text files (.txt, .md, .csv)
        - PDFs (.pdf)
        - Images (.png, .jpg, .jpeg) - extracted via OCR/vision models
        - Audio files (.mp3, .wav) - transcribed to text
        - Code files (.py, .js, .ts, etc.) - parsed for structure and content
        - Office documents (.docx, .pptx)

    Args:
        data: The latest version of the data. ``data_id`` names the document, so the
            replacement may carry any filename; the new name lands on the row. Can be:
            - Single text string: "Your text content here"
            - DLT resource or source: replaces a DLT source manifest under the same
              source name (the whole source is re-ingested and re-cognified)
            - Absolute file path: "/path/to/document.pdf"
            - File URL: "file:///absolute/path/to/document.pdf" or "file://relative/path.txt"
            - S3 path: "s3://my-bucket/documents/file.pdf"
            - Binary file object: open("file.txt", "rb")
            - A directory path, or a list of any of the above: a batch
        dataset_id: UUID of the dataset holding the document (required).
        data_id: UUID of the document to replace (current or pre-fork). Optional
            for a local file or an upload, whose document is inferred from the
            path or filename; required for raw text and for a file that was
            renamed or moved. Only with a single input.
        user: User object for authentication and permissions. Uses default user if None.
              Default user: "default_user@example.com" (created automatically on first use).
              Users can only access datasets they have permissions for.
        node_set: Optional list of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        vector_db_config: Optional configuration for vector database (for custom setups).
                 Chunk-level incremental updates do not support per-call config
                 forwarding: when provided, the update runs full ingestion
                 instead (a warning is logged). For incremental updates,
                 configure stores through environment settings and the
                 dataset-context database routing system.
        graph_db_config: Optional configuration for graph database (for custom setups).
                 Same routing note as vector_db_config.
        chunk_level_diff: When True (default), diff the new content against the stored
                 processed text and replace only the chunks the edit touched — unaffected
                 chunks keep their nodes, entities, and summaries. Falls back to the full
                 rebuild (memory dropped, row refreshed by a pinned re-add, cognify) when
                 chunk-level preconditions are
                 not met (first ingestion, non-text content, unverified graph adapter,
                 a code file or a DLT source manifest — those routes keep no chunks).
                 Permission errors always propagate and never trigger the fallback.
        chunker: Chunking strategy. Must match the one that built the document's stored
                 chunks — a mismatch is refused (and falls back) rather than surfacing
                 as a tiling failure. Chunk-level path only.
        policy: Decides which chunks exist after the edit and what happens to the old
                 ones. Replaceable without touching storage or update orchestration.
                 Chunk-level path only; not exposed on the HTTP route.

    Returns:
        For a single input, one dict on every path (schema: ``UpdateResult``), a
        superset of the chunk-level summary returned before:
            - ``status``: "incremental" (chunks replaced), "unchanged" (no content
              change), "full_rebuild" (memory dropped and rebuilt from the new
              content) or "failed"
              (the rebuild's cognify run errored; ``error`` says why, and the call
              can be retried).
            - ``regions``, ``deleted_chunks``, ``added_chunks``, ``reused_chunks``,
              ``kept_chunks``, ``reindexed_chunks``, ``total_chunks``: the chunk-level
              counters; None on a rebuild, which has no diff.
            - ``data_id``, ``dataset_id``: the document, the handle to retry with.
            - ``duration_seconds``: wall-clock time of the update.
            - ``pipeline_run_id``: the run to inspect; None for a no-op.
            - ``fallback``: set on every rebuild, its ``reason`` and ``detail`` naming
              why the chunk-level path did not run — the caller switched it off, an
              unsupported parameter, or one of the engine's refusals.
            - ``error``: ``error_class`` and ``message`` when ``status`` is "failed".
        For a list of several inputs or a directory, a dict with schema
        ``UpdateBatchResult``:
            - ``status``: "completed" (nothing failed), "partial" or "failed" (every
              document failed).
            - ``total``, ``updated`` (incremental or full_rebuild), ``unchanged``,
              ``failed``: the counts.
            - ``dataset_id``, ``duration_seconds`` of the whole batch.
            - ``results``: one ``UpdateResult`` per input, in input order.
    """
    if isinstance(data, UUID):
        raise TypeError(
            "update() takes the new content first: update(data, dataset_id, data_id=...)"
        )

    # Route to the remote instance when connected via serve(). This must come
    # before any local work: the paths below resolve the LOCAL default user and
    # write locally, which against a remote dataset id fails with
    # "Dataset not found" while the remote document stays untouched.
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        dropped = [
            name
            for name, value, default in (
                ("vector_db_config", vector_db_config, None),
                ("graph_db_config", graph_db_config, None),
                ("preferred_loaders", preferred_loaders, None),
                ("graph_model", graph_model, KnowledgeGraph),
                ("custom_prompt", custom_prompt, None),
                ("chunker", chunker, TextChunker),
                ("policy", policy, DEFAULT_CHUNK_POLICY),
            )
            if value is not default
        ]
        if dropped:
            logger.warning(
                "update() is proxied to the remote instance; PATCH /api/v1/update has no "
                "slot for %s — the server applies its own configuration",
                ", ".join(dropped),
            )
        return await client.update(
            data=data,
            dataset_id=dataset_id,
            data_id=data_id,
            node_set=node_set,
            chunk_level_diff=chunk_level_diff,
        )

    started = perf_counter()
    if not user:
        user = await get_default_user()

    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.modules.ingestion.identify_by_origin import expand_directories

    given = data if isinstance(data, list) else [data]
    items = expand_directories(given)
    if not items:
        raise IngestionError("update() got no documents to update.")
    if data_id is not None and len(items) != 1:
        raise IngestionError(
            f"data_id names one document; got {len(items)} inputs with it. Pass a single "
            "input, or per-document ids as DataItem(data=..., data_id=...)."
        )
    # Every target is known before the first document is touched: an input
    # that names no document fails the call with nothing written.
    targets = await _resolve_targets(items, dataset_id, data_id, user)

    options = {
        "dataset_id": dataset_id,
        "user": user,
        "node_set": node_set,
        "vector_db_config": vector_db_config,
        "graph_db_config": graph_db_config,
        "preferred_loaders": preferred_loaders,
        "incremental_loading": incremental_loading,
        "data_cache": data_cache,
        "chunk_level_diff": chunk_level_diff,
        "graph_model": graph_model,
        "custom_prompt": custom_prompt,
        "chunker": chunker,
        "policy": policy,
    }
    # A one-item list is that one document, as it always was (the shape the
    # HTTP router sends); a list of several, or a directory, is a batch.
    if len(given) == 1 and items == given:
        return await _update_one(items[0], targets[0], **options)

    results = []
    for item, pinned_id in zip(items, targets):
        item_started = perf_counter()
        try:
            results.append(await _update_one(item, pinned_id, **options))
        except Exception as error:
            # One document's failure must not lose the others' results: it is
            # recorded, with its id, for the caller to retry on its own.
            logger.exception("update of document %s in dataset %s failed", pinned_id, dataset_id)
            results.append(
                UpdateResult(
                    status="failed",
                    data_id=pinned_id,
                    dataset_id=dataset_id,
                    duration_seconds=round(perf_counter() - item_started, 3),
                    error=UpdateError(error_class=type(error).__name__, message=str(error)),
                ).model_dump()
            )
    return UpdateBatchResult.of(
        results, dataset_id, round(perf_counter() - started, 3)
    ).model_dump()


def _describe_input(item) -> str:
    """An input as a person would name it, for the refusal message."""
    if isinstance(item, str):
        return f"text {item[:40]!r}{'…' if len(item) > 40 else ''}"
    return f"{type(item).__name__} input"


async def _resolve_targets(
    items: list, dataset_id: UUID, data_id: UUID | None, user: User
) -> list[UUID]:
    """The document each input replaces, in input order.

    An explicit id (``data_id``, or ``DataItem.data_id``) is resolved exactly
    or through the recorded pre-fork ``legacy_id``; a stale or mistyped id is
    a caller error, never a create. Everything else is inferred from the
    input's origin with one batched lookup; whatever cannot be matched is
    refused together, with its reason, so the caller fixes the call once.
    """
    from cognee.api.v1.exceptions import UpdateTargetNotFoundError, UpdateTargetNotInferredError
    from cognee.modules.data.methods import resolve_data_id
    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.modules.ingestion.identify_by_origin import Origin, find_by_origin, origin_of
    from cognee.tasks.ingestion.data_item import DataItem

    targets: list[UUID | None] = [None] * len(items)
    unresolved: list[dict] = []
    inferred: list[tuple[int, Origin]] = []
    for index, item in enumerate(items):
        explicit = data_id
        if explicit is None and isinstance(item, DataItem):
            explicit = item.data_id
        if explicit is not None:
            resolved = await resolve_data_id(dataset_id, explicit)
            if resolved is None:
                raise UpdateTargetNotFoundError(data_id=explicit, dataset_id=dataset_id)
            targets[index] = resolved
            continue
        content = item.data if isinstance(item, DataItem) else item
        origin = await origin_of(content)
        if origin is None:
            reason = (
                "raw text has no origin to match"
                if isinstance(content, str)
                else "not a local file or an upload, so it has no origin to match"
            )
            unresolved.append({"input": _describe_input(content), "reason": reason})
        else:
            inferred.append((index, origin))

    matches = await find_by_origin([origin for _, origin in inferred], user, dataset_id)
    for index, origin in inferred:
        found = matches[origin]
        if len(found) == 1:
            targets[index] = found[0].data_id
        elif not found:
            unresolved.append(
                {"input": origin.label, "reason": "no document in the dataset came from it"}
            )
        else:
            unresolved.append(
                {"input": origin.label, "reason": f"{len(found)} documents came from it"}
            )
    if unresolved:
        raise UpdateTargetNotInferredError(unresolved, dataset_id)

    resolved = [target for target in targets if target is not None]
    seen: dict[UUID, int] = {}
    for index, target in enumerate(resolved):
        if target in seen:
            raise IngestionError(
                f"inputs {seen[target] + 1} and {index + 1} both target document {target}; "
                "a batch updates each document once."
            )
        seen[target] = index
    return resolved


async def _update_one(
    data,
    pinned_id: UUID,
    dataset_id: UUID,
    user: User,
    node_set,
    vector_db_config,
    graph_db_config,
    preferred_loaders,
    incremental_loading,
    data_cache,
    chunk_level_diff,
    graph_model,
    custom_prompt,
    chunker,
    policy,
) -> dict:
    """Replace the document ``pinned_id`` with ``data``; the single-document path."""
    started = perf_counter()

    from cognee.modules.data.methods import reset_data_pipeline_status
    from cognee.tasks.ingestion.data_item import DataItem
    from cognee.tasks.ingestion.resolve_dlt_sources import check_dlt_replacement, is_dlt_input

    # Why the chunk-level path is not taken, if it is not. Every full rebuild
    # names its cause in the result, so an update that took far longer than
    # usual explains itself instead of leaving the reason in the server log.
    fallback = _full_rebuild_reason(
        chunk_level_diff,
        data,
        node_set,
        graph_model,
        custom_prompt,
        vector_db_config,
        graph_db_config,
    )
    if fallback is not None:
        logger.warning("%s; running full update", fallback[1])

    if fallback is None:
        # Chunk-level incremental path: diff the new text against the stored
        # processed text, replace only the affected chunks — the Data row is
        # updated in place, so the id trivially survives. Falls through to
        # the full flow when its preconditions aren't met (first ingestion,
        # non-text content, stored chunks unavailable). Permission errors
        # propagate — they must never trigger the fallback.
        try:
            summary = await incremental_update(
                data_id=pinned_id,
                data=data,
                dataset_id=dataset_id,
                user=user,
                node_set=node_set,
                preferred_loaders=preferred_loaders,
                graph_model=graph_model,
                custom_prompt=custom_prompt,
                chunker=chunker,
                policy=policy,
            )
        except IncrementalUpdateNotPossible as refusal:
            # The reason is a structured field, not just prose: an unsupported
            # chunker and a first ingestion produce the same sentence otherwise,
            # so a permanent misconfiguration is indistinguishable from a
            # one-off in the logs.
            logger.warning(
                "chunk-level update not possible (%s); running full update",
                refusal,
                extra={"refusal_reason": refusal.reason.value},
            )
            fallback = (refusal.reason, str(refusal))
        else:
            return UpdateResult(
                status=summary["status"],
                regions=summary["regions"],
                deleted_chunks=summary["deleted_chunks"],
                added_chunks=summary["added_chunks"],
                reused_chunks=summary["reused_chunks"],
                kept_chunks=summary["kept_chunks"],
                reindexed_chunks=summary["reindexed_chunks"],
                total_chunks=summary["total_chunks"],
                data_id=pinned_id,
                dataset_id=dataset_id,
                duration_seconds=round(perf_counter() - started, 3),
                pipeline_run_id=summary["pipeline_run_id"],
            ).model_dump()

    # The rebuild re-cognifies the whole document. It keeps the chunk budget
    # the stored chunks record so the document's granularity survives the
    # rebuild, whichever way the rebuild was decided; None (no baseline, or a
    # recorded budget the current provider cannot take) means the current
    # default.
    fallback_chunk_size = await recorded_chunk_budget(pinned_id, dataset_id, user)

    # A dlt source's identity is decided by the resolver, not by the pinned
    # id, so a replacement the re-add would refuse is refused now, before the
    # document's memory is dropped.
    replacement = data.data if isinstance(data, DataItem) else data
    if is_dlt_input(replacement):
        from cognee.modules.data.methods import get_authorized_dataset

        dataset = await get_authorized_dataset(user, dataset_id, "write")
        await check_dlt_replacement(replacement, pinned_id, dataset.name, user)

    # The rebuild never deletes the document. Its memory (graph nodes, edges,
    # vectors) is dropped while the row and its stored files stay; the pinned
    # re-add then refreshes the row in place with the new content, name and
    # metadata, and cognify rebuilds the graph. A re-add that fails leaves a
    # document with no graph that the next cognify() rebuilds from its stored
    # content, not a document that is gone. The row keeps its id, owner and
    # pre-fork legacy id by construction.
    await forget(data_id=pinned_id, dataset_id=dataset_id, memory_only=True, user=user)
    # forget() clears the cognify stamp; the add stamp must go too, or the
    # pinned re-add is skipped as already added and the row is never refreshed.
    await reset_data_pipeline_status(pinned_id, dataset_id)

    if isinstance(data, DataItem):
        data.data_id = pinned_id
        pinned_item = data
    else:
        pinned_item = DataItem(data=data, data_id=pinned_id)

    await add(
        data=pinned_item,
        dataset_id=dataset_id,
        user=user,
        node_set=node_set,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        preferred_loaders=preferred_loaders,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
    )

    cognify_runs = await cognify(
        datasets=[dataset_id],
        user=user,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
        graph_model=graph_model,
        custom_prompt=custom_prompt,
        chunk_size=fallback_chunk_size,
        # An errored run is reported as a failed result, not raised: the
        # caller gets the document id and the error to retry this one update.
        raise_on_error=False,
    )

    errored = get_errored_run_info(cognify_runs)
    run = errored or next(iter(cognify_runs.values()))
    return UpdateResult(
        status="failed" if errored else "full_rebuild",
        data_id=pinned_id,
        dataset_id=dataset_id,
        duration_seconds=round(perf_counter() - started, 3),
        pipeline_run_id=run.pipeline_run_id,
        fallback=Fallback(reason=fallback[0], detail=fallback[1]),
        error=UpdateError(error_class=errored.error_class, message=errored.error_message)
        if errored
        else None,
    ).model_dump()


def _full_rebuild_reason(
    chunk_level_diff: bool,
    data,
    node_set,
    graph_model,
    custom_prompt,
    vector_db_config,
    graph_db_config,
) -> tuple[RefusalReason, str] | None:
    """Decide, before the engine is consulted, whether the full rebuild must run.

    Returns the reason and a sentence for the caller, or None when the
    chunk-level path may be attempted. Each cause is a limitation of the
    chunk-level engine: it reconciles chunks, not document metadata; the
    baseline does not persist the model or prompt that produced its graph, so
    a different one applied to fresh chunks only would mix extraction schemas
    inside one document; and it resolves its stores through the dataset
    context, not per-call config dicts, so running it with those would read
    and write the default stores while the caller's stores never see the edit.
    """
    from cognee.tasks.ingestion.data_item import DataItem

    if not chunk_level_diff:
        return RefusalReason.DISABLED, "chunk_level_diff=False was requested"

    data_item_changes_metadata = isinstance(data, DataItem) and (
        data.label is not None or data.external_metadata is not None
    )
    if node_set or data_item_changes_metadata:
        return (
            RefusalReason.UNSUPPORTED_METADATA,
            "chunk-level update does not reconcile node_set or document metadata",
        )
    if graph_model is not KnowledgeGraph or custom_prompt is not None:
        return (
            RefusalReason.CUSTOM_EXTRACTION_CONFIG,
            "chunk-level update supports only the default graph model and prompt",
        )
    if vector_db_config is not None or graph_db_config is not None:
        return (
            RefusalReason.PER_CALL_DB_CONFIG,
            (
                "chunk-level update does not take per-call vector_db_config/graph_db_config; "
                "configure stores through environment settings and the dataset-context routing"
            ),
        )
    return None
