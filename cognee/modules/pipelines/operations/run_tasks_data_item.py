"""
Data item processing functions for pipeline operations.

This module contains reusable functions for processing individual data items
within pipeline operations, supporting both incremental and regular processing modes.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules import ingestion
from cognee.modules.data.models import Data, Dataset
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.models.DataItemStatus import (
    DataItemStatus,
    is_data_item_completed,
)
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunProgress,
    PipelineRunYield,
)
from cognee.modules.pipelines.operations.run_tasks_with_telemetry import run_tasks_with_telemetry
from cognee.modules.pipelines.queues.pipeline_run_info_queues import push_to_queue
from cognee.modules.provenance.edge_evidence.persistence import flush_context_provenance
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion.carried_source import publish_carried_source
from cognee.tasks.ingestion.save_data_item_to_storage import (
    save_data_item_to_storage_detailed,
)

from ..tasks.task import Task

logger = get_logger("run_tasks_data_item")


def _push_stage_progress(
    yielded: PipelineRunYield,
    ctx: PipelineContext | None,
    tasks: list[Task],
    pipeline_run_id: str,
    progress_state: dict[str, Any] | None = None,
) -> None:
    """Turn one intermediate PipelineRunYield into a PipelineRunProgress event
    and push it to this run's queue, so backgrounded runs surface per-stage
    progress over the /subscribe WebSocket instead of only the terminal event.

    ``ctx.task_sequence`` (populated by handle_task in run_tasks_base.py as
    each task executes) already names which stage just ran and its 1-based
    position — no separate stage-tracking state needed here. push_to_queue
    is a no-op when no queue exists for this pipeline_run_id (blocking mode),
    so this is safe to call unconditionally.

    ``progress_state`` (when given) is a dict owned by run_tasks.py, holding
    only ``current_stage`` — this function is its sole writer. run_tasks.py
    reads it back (alongside its own separately-tracked completed_items/
    total_items counters) when calling log_pipeline_run_progress, so /status
    (not just the WebSocket) can surface the current stage too.
    """
    current_stage = ctx.task_sequence[-1] if ctx and ctx.task_sequence else None
    stage_index = len(ctx.task_sequence) if ctx else None
    if progress_state is not None:
        progress_state["current_stage"] = current_stage
    # pipeline_run_id is sourced from the function's own parameter (the queue
    # key), not yielded.pipeline_run_id, so the event body can never disagree
    # with the subscription it's delivered under.
    push_to_queue(
        pipeline_run_id,
        PipelineRunProgress(
            pipeline_run_id=pipeline_run_id,
            dataset_id=yielded.dataset_id,
            dataset_name=yielded.dataset_name,
            current_stage=current_stage,
            stage_index=stage_index,
            stage_total=len(tasks),
        ),
    )


async def _drain_item_events(
    events: AsyncGenerator[Any, None],
    ctx: PipelineContext | None,
    tasks: list[Task],
    pipeline_run_id: str,
    progress_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Consume one run_tasks_data_item_incremental/_regular generator to its end.

    Every intermediate ``PipelineRunYield`` is forwarded to stage-progress
    tracking; the one non-yield item — a ``{"run_info": ..., "data_id": ...}``
    dict — is the item's final result, returned once the generator is exhausted.

    If progress tracking raises mid-item, ``aclosing`` closes the generator
    right away, so the item's telemetry reports a terminal event now rather than
    whenever the abandoned generator is garbage-collected.
    """
    result = None
    async with aclosing(events) as stream:
        async for item in stream:
            if isinstance(item, PipelineRunYield):
                _push_stage_progress(item, ctx, tasks, pipeline_run_id, progress_state)
            else:
                result = item
    return result


async def run_tasks_data_item_incremental(
    data_item: Any,
    dataset: Dataset,
    tasks: list[Task],
    pipeline_name: str,
    pipeline_id: str,
    pipeline_run_id: str,
    ctx: PipelineContext | None,
    user: User,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Process a single data item with incremental loading support.

    This function handles incremental processing by checking if the data item
    has already been processed for the given pipeline and dataset. If it has,
    it skips processing and returns a completion status.

    Args:
        data_item: The data item to process
        dataset: The dataset containing the data item
        tasks: List of tasks to execute on the data item
        pipeline_name: Name of the pipeline
        pipeline_id: Unique identifier for the pipeline
        pipeline_run_id: Unique identifier for this pipeline run
        context: Optional context dictionary
        user: User performing the operation

    Yields:
        Dict containing run_info and data_id for each processing step
    """
    db_engine = get_relational_engine()

    # If incremental_loading of data is set to True don't process documents already processed by pipeline
    # If data is being added to Cognee for the first time resolve the id of the data.
    #
    # Every session below is a fresh connection on the cloud pods (NullPool
    # over Neon: TCP + TLS + SCRAM per session, ~14 ms of CPU before any
    # latency), and this wrapper runs once PER ITEM — so the pre-check resolves
    # the row and reads its pipeline_status in ONE lookup instead of
    # identify()-then-select-by-id, and the post-run status write below
    # re-resolves fresh content inside the session that records the status.
    content_hash = None
    data_point = None
    if not isinstance(data_item, Data):
        # If the DataItem carries a stable data_id (e.g. from DLT), prefer it
        # over the content lookup so lookups stay consistent.
        from cognee.modules.data.methods import resolve_data_id
        from cognee.tasks.ingestion.data_item import DataItem as DataItemType

        if isinstance(data_item, DataItemType) and data_item.data_id is not None:
            # Resolve the pin as ingest_data does: a legacy id names the data
            # item it was forked into, which is where the content is stored.
            pin = data_item.data_id
            data_id = await resolve_data_id(dataset.id, pin) or pin
            async with db_engine.get_async_session() as session:
                data_point = (
                    await session.execute(select(Data).filter(Data.id == data_id))
                ).scalar_one_or_none()
        else:
            stored = await save_data_item_to_storage_detailed(data_item)

            # For payloads whose bytes passed through this process the hash was
            # computed at save time — re-reading the object just written (over
            # S3 a HEAD plus a full GET) to recompute a byte-identical hash was
            # the most expensive thing this wrapper did, and it did it on the
            # event loop via the sync run_sync bridge. Items cognee did not
            # write (an s3:// URL, a local path) must still be read — while the
            # file is open, since the metadata read consumes the stream.
            if stored.metadata is None:
                async with open_data_file(stored.file_path) as file:
                    stored.metadata = await ingestion.classify(file).aget_metadata()

            content_hash = stored.metadata["content_hash"]

            # Dataset-scoped content lookup: the existing row (its id and
            # pipeline_status) or None for content this dataset has not
            # seen (ingestion mints the id).
            data_point = await ingestion.identify_data_by_hash(content_hash, user, dataset.id)
            data_id = data_point.id if data_point is not None else None

            # Hand the storage work already paid for to ``ingest_data``, which
            # otherwise re-uploads and re-hashes the very same item.
            publish_carried_source(ctx, data_item, stored)
    else:
        # If data was already processed by Cognee get data id
        data_id = data_item.id
        async with db_engine.get_async_session() as session:
            data_point = (
                await session.execute(select(Data).filter(Data.id == data_id))
            ).scalar_one_or_none()

    # Check pipeline status, if Data already processed for pipeline before skip current processing
    if data_point:
        status_for_pipeline = (data_point.pipeline_status or {}).get(pipeline_name) or {}
        if is_data_item_completed(status_for_pipeline.get(str(dataset.id))):
            yield {
                "run_info": PipelineRunAlreadyCompleted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                ),
                "data_id": data_id,
                **_source_fields(data_point, data_item),
            }
            return

    try:
        # Process data based on data_item and list of tasks. aclosing closes
        # the telemetry generator as soon as this one is closed early.
        async with aclosing(
            run_tasks_with_telemetry(
                tasks=tasks,
                data=[data_item],
                user=user,
                pipeline_name=pipeline_id,
                ctx=ctx,
            )
        ) as results:
            async for result in results:
                yield PipelineRunYield(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    payload=result,
                )

        # Update pipeline status for Data element. Fresh content had no row at
        # the pre-check (data_id None); ingestion has created it since — resolve
        # the row it was given, in the same session that records the status.
        async with db_engine.get_async_session() as session:
            if data_id is not None:
                data_point = (
                    await session.execute(select(Data).filter(Data.id == data_id))
                ).scalar_one_or_none()
            elif content_hash is not None:
                data_point = await ingestion.identify_data_by_hash(
                    content_hash, user, dataset.id, session=session
                )
                data_id = data_point.id if data_point is not None else None
            else:
                data_point = None
            if data_point is not None:
                status_for_pipeline = data_point.pipeline_status.setdefault(pipeline_name, {})
                status_for_pipeline[str(dataset.id)] = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
                await session.merge(data_point)
                await session.commit()

        yield {
            "run_info": PipelineRunCompleted(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
            ),
            "data_id": data_id,
            **_source_fields(data_point, data_item),
        }

    except Exception as error:
        # Temporarily swallow error and try to process rest of documents first, then re-raise error at end of data ingestion pipeline
        logger.error(
            f"Exception caught while processing data: {error}.\n Data processing failed for data item: {data_item}."
        )
        from cognee.modules.operations import scrub_error_message

        yield {
            "run_info": PipelineRunErrored(
                pipeline_run_id=pipeline_run_id,
                payload=repr(error),
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                error_class=type(error).__name__,
                error_message=scrub_error_message(error),
            ),
            # In-memory handle to the root cause, so run_tasks can record and
            # surface WHAT failed even on the non-raising path — otherwise the
            # run ends as a generic "Pipeline run failed. Data item could not
            # be processed." with a NULL error column.
            "error": error,
            "data_id": data_id,
            **_source_fields(data_point, data_item),
        }

        if os.getenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true").lower() == "true":
            raise


def _source_fields(data_point: Data | None, data_item: Any) -> dict[str, Any]:
    """Name, location and label for an item's per-item result entry.

    They let a caller tell which input an id belongs to (a folder yields one
    entry per file). Read from the stored data item when there is one, else
    from the input itself, as for an item that failed before it was stored.
    """
    if data_point is not None:
        return {
            "data_name": data_point.name,
            "data_location": _stored_source_uri(data_point),
            "data_label": data_point.label,
        }
    return _input_source_fields(data_item)


def _stored_source_uri(data_point: Data) -> str | None:
    """Where the caller's content came from, as ingest_data recorded it.

    Raw text has no source of its own, so it has none; cognee's stored copy is
    deliberately not reported as one.
    """
    # Data items stored by older versions can hold any JSON here, e.g. a string.
    external_metadata = data_point.external_metadata
    if not isinstance(external_metadata, dict):
        return None
    cognee_metadata = external_metadata.get("_cognee")
    if not isinstance(cognee_metadata, dict):
        return None
    source_uri = cognee_metadata.get("source_uri")
    return source_uri if isinstance(source_uri, str) else None


def _input_source_fields(data_item: Any) -> dict[str, Any]:
    """Name, location and label read from the input, for an item not stored yet."""
    from cognee.tasks.ingestion.data_item import DataItem as DataItemType
    from cognee.tasks.ingestion.ingest_data import _source_uri_from_input

    location = _source_uri_from_input(data_item)
    name = Path(unquote(urlparse(location).path)).stem if location else None
    return {
        "data_name": name or None,
        "data_location": location,
        "data_label": data_item.label if isinstance(data_item, DataItemType) else None,
    }


async def run_tasks_data_item_regular(
    data_item: Any,
    dataset: Dataset,
    tasks: list[Task],
    pipeline_id: str,
    pipeline_run_id: str,
    ctx: PipelineContext | None,
    user: User,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Process a single data item in regular (non-incremental) mode.

    This function processes a data item without checking for previous processing
    status, executing all tasks on the data item.

    Args:
        data_item: The data item to process
        dataset: The dataset containing the data item
        tasks: List of tasks to execute on the data item
        pipeline_id: Unique identifier for the pipeline
        pipeline_run_id: Unique identifier for this pipeline run
        context: Optional context dictionary
        user: User performing the operation

    Yields:
        Dict containing run_info for each processing step
    """
    # Process data based on data_item and list of tasks. aclosing closes the
    # telemetry generator as soon as this one is closed early.
    async with aclosing(
        run_tasks_with_telemetry(
            tasks=tasks,
            data=[data_item],
            user=user,
            pipeline_name=pipeline_id,
            ctx=ctx,
        )
    ) as results:
        async for result in results:
            yield PipelineRunYield(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=result,
            )

    yield {
        "run_info": PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
        )
    }


async def run_tasks_data_item(
    data_item: Any,
    dataset: Dataset,
    tasks: list[Task],
    pipeline_name: str,
    pipeline_id: str,
    pipeline_run_id: str,
    ctx: PipelineContext | None,
    user: User,
    incremental_loading: bool,
    data_cache: bool,
    progress_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Process a single data item, choosing between incremental and regular processing.

    This is the main entry point for data item processing that delegates to either
    incremental or regular processing based on the data_cache/incremental_loading flags.

    Args:
        data_item: The data item to process
        dataset: The dataset containing the data item
        tasks: List of tasks to execute on the data item
        pipeline_name: Name of the pipeline
        pipeline_id: Unique identifier for the pipeline
        pipeline_run_id: Unique identifier for this pipeline run
        context: Optional context dictionary
        user: User performing the operation
        incremental_loading: Whether to use incremental processing
        data_cache: Whether to use incremental processing (data caching)
        progress_state: Optional shared dict (owned by run_tasks.py) that this
            item's stage-progress ticks record ``current_stage`` into, so the
            item-level progress log picks up the most recent stage name.

    Returns:
        Dict containing the final processing result, or None if processing was skipped
    """
    # Result can be PipelineRunAlreadyCompleted when data item is skipped,
    # PipelineRunCompleted when processing was successful, or PipelineRunErrored
    # if there were issues — _drain_item_events pulls that final result out of
    # the generator while forwarding every intermediate stage tick.
    if data_cache or incremental_loading:
        events = run_tasks_data_item_incremental(
            data_item=data_item,
            dataset=dataset,
            tasks=tasks,
            pipeline_name=pipeline_name,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            ctx=ctx,
            user=user,
        )
    else:
        events = run_tasks_data_item_regular(
            data_item=data_item,
            dataset=dataset,
            tasks=tasks,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            ctx=ctx,
            user=user,
        )

    try:
        result = await _drain_item_events(events, ctx, tasks, pipeline_run_id, progress_state)
    except Exception:
        # Preserve the original pipeline exception if flushing already-written
        # edge evidence also fails; rollback still has graph-native run refs.
        try:
            await flush_context_provenance(ctx)
        except Exception:
            logger.exception(
                "Failed to persist provenance for an errored data item",
            )
        raise

    await flush_context_provenance(ctx)
    return result
