import os

import asyncio
from functools import wraps
from typing import Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select

import cognee.modules.ingestion as ingestion
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.data.models import Data
from cognee.modules.pipelines.exceptions import PipelineItemFailure, PipelineRunFailedError
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
    PipelineRunYield,
)
from cognee.modules.pipelines.operations import (
    log_pipeline_run_complete,
    log_pipeline_run_error,
    log_pipeline_run_start,
)
from cognee.modules.pipelines.operations.run_tasks_distributed import run_tasks_distributed
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from cognee.tasks.ingestion import resolve_data_directories, save_data_item_to_storage

from ..tasks.task import Task
from .worker_pipeline import _ErroredItem, _OriginRef, run_worker_pipeline


logger = get_logger("run_tasks(tasks: [Task], data)")


def override_run_tasks(new_gen):
    """Decorator factory that conditionally routes pipeline execution between distributed and local modes."""

    def decorator(original_gen):
        """Wraps the original generator to add distributed execution routing."""

        @wraps(original_gen)
        async def wrapper(*args, distributed=None, **kwargs):
            """Routes execution based on COGNEE_DISTRIBUTED env var or explicit distributed parameter."""
            default_distributed_value = os.getenv("COGNEE_DISTRIBUTED", "False").lower() == "true"
            distributed = default_distributed_value if distributed is None else distributed

            if distributed:
                async for run_info in new_gen(*args, **kwargs):
                    yield run_info
            else:
                async for run_info in original_gen(*args, **kwargs):
                    yield run_info

        return wrapper

    return decorator


async def _resolve_data_id(data_item: Any, user: User) -> Any:
    """Compute the stable ``data_id`` used to look up per-item processing
    status. Lifted out of run_tasks_data_item_incremental so we can run it
    in parallel before feeding the worker pipeline."""
    if isinstance(data_item, Data):
        return data_item.id

    from cognee.tasks.ingestion.data_item import DataItem as DataItemType

    if isinstance(data_item, DataItemType) and data_item.data_id is not None:
        return data_item.data_id

    file_path = await save_data_item_to_storage(data_item)
    async with open_data_file(file_path) as file:
        classified_data = ingestion.classify(file)
        return await ingestion.identify(classified_data, user)


async def _is_already_completed(data_id, pipeline_name: str, dataset_id) -> bool:
    """Check if a data item has already been processed for this pipeline in incremental mode."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
    if not data_point:
        return False
    status = data_point.pipeline_status or {}
    return (
        status.get(pipeline_name, {}).get(str(dataset_id))
        == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
    )


async def _mark_completed(data_id, pipeline_name: str, dataset_id) -> None:
    """Mark a data item as completed for a pipeline in incremental mode."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # SELECT ... FOR UPDATE so concurrent pipeline runs marking the same
        # Data row don't lose each other's status entries under last-write-wins.
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id).with_for_update())
        ).scalar_one_or_none()
        if not data_point:
            return
        status = data_point.pipeline_status or {}
        status_for_pipeline = status.setdefault(pipeline_name, {})
        status_for_pipeline[str(dataset_id)] = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
        data_point.pipeline_status = status
        await session.merge(data_point)
        await session.commit()


async def _preflight(
    data: List[Any],
    user: User,
    dataset,
    pipeline_name: str,
    incremental_loading: bool,
    data_per_batch: int,
) -> Tuple[List[Tuple[Any, Any]], List[Any]]:
    """Resolve data_ids and (for incremental mode) filter out items that have
    already been processed.

    Returns ``(items_to_process, already_completed_ids)`` where each item in
    ``items_to_process`` is ``(data_item, data_id_or_None)``.

    Concurrency for the ``asyncio.gather`` fan-outs below is capped at
    ``data_per_batch`` via a semaphore so large input lists do not exhaust the
    relational DB connection pool when resolving / checking many items in
    parallel.
    """
    if not incremental_loading:
        return [(item, None) for item in data], []

    sem = asyncio.Semaphore(data_per_batch)

    async def _bounded_resolve(item):
        """Resolve data_id with semaphore to limit concurrency."""
        async with sem:
            return await _resolve_data_id(item, user)

    data_ids = await asyncio.gather(*[_bounded_resolve(item) for item in data])

    async def _bounded_check(did):
        """Check completion status with semaphore to limit concurrency."""
        async with sem:
            return await _is_already_completed(did, pipeline_name, dataset.id)

    completion_flags = await asyncio.gather(*[_bounded_check(did) for did in data_ids])

    to_process: List[Tuple[Any, Any]] = []
    already_done: List[Any] = []
    for item, did, done in zip(data, data_ids, completion_flags):
        if done:
            already_done.append(did)
        else:
            to_process.append((item, did))
    return to_process, already_done


def _pipeline_telemetry_props(pipeline_name: str, user: User) -> dict:
    """Build telemetry properties dict with pipeline name, version, and tenant info."""
    return {
        "pipeline_name": str(pipeline_name),
        "cognee_version": cognee_version,
        "tenant_id": str(user.tenant_id) if user and user.tenant_id else "Single User Tenant",
    }


def _safe_send_telemetry(event_name: str, user_id, additional_properties: dict) -> None:
    """Emit a telemetry event without letting a telemetry outage abort the
    pipeline. Failures are logged and swallowed."""
    try:
        send_telemetry(event_name, user_id, additional_properties=additional_properties)
    except Exception:
        logger.warning("Telemetry emission failed for %s", event_name, exc_info=True)


@override_run_tasks(run_tasks_distributed)
async def run_tasks(
    tasks: List[Task],
    dataset_id: UUID,
    data: Optional[List[Any]] = None,
    user: Optional[User] = None,
    pipeline_name: str = "unknown_pipeline",
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    extras: Optional[dict] = None,
):
    """Run a pipeline once for an entire dataset: a single shared worker
    pipeline absorbs all data items, with per-task ``num_workers`` (default
    ``data_per_batch`` when the task allows reordering) providing input
    parallelism."""
    if data_per_batch <= 0:
        raise ValueError(f"data_per_batch must be > 0, got {data_per_batch}")

    if not user:
        user = await get_default_user()

    async with get_relational_engine().get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset.id, data)
    pipeline_run_id = pipeline_run.pipeline_run_id

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    _safe_send_telemetry(
        "Pipeline Run Started",
        user.id,
        additional_properties=_pipeline_telemetry_props(pipeline_name, user),
    )

    # Note: Setting of global context has to be done after yielding PipelineRunStarted
    # due to running in background mode requiring the pipeline run started yield.
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        try:
            if not isinstance(data, list):
                data = [data]

            if incremental_loading:
                data = await resolve_data_directories(data)

            to_process, already_done_ids = await _preflight(
                data=data,
                user=user,
                dataset=dataset,
                pipeline_name=pipeline_name,
                incremental_loading=incremental_loading,
                data_per_batch=data_per_batch,
            )

            results: list = []
            for did in already_done_ids:
                results.append(
                    {
                        "run_info": PipelineRunAlreadyCompleted(
                            pipeline_run_id=pipeline_run_id,
                            dataset_id=dataset.id,
                            dataset_name=dataset.name,
                        ),
                        "data_id": did,
                    }
                )

            ctx = PipelineContext(
                user=user,
                data_item=None,  # set per-call inside workers via dataclasses.replace
                dataset=dataset,
                pipeline_name=pipeline_name,
                extras=extras if isinstance(extras, dict) else {},
            )

            # First task expects a list of Data items (cognee convention),
            # so wrap each item as [item] entering the pipeline. The
            # ``origin`` (second tuple element) is a fresh ``_OriginRef``
            # wrapper per pipeline item — each wrapper has a distinct
            # ``id()`` even when the caller deduplicated by reusing the same
            # Python object across multiple inputs, so ``origin_state`` keys
            # never alias. Workers unwrap ``ref.item`` to populate
            # ``ctx.data_item`` with the original Data downstream.
            pipeline_inputs = [([item], _OriginRef(item)) for item, _ in to_process]

            # Track per-origin processing state keyed by id(ref) where ``ref``
            # is the same ``_OriginRef`` instance pushed into the pipeline. We
            # need identity-based lookup (Data items may be value-equal
            # duplicates, so __hash__/__eq__ won't do); the ``_OriginRef``
            # wrapper guarantees a distinct ``id()`` per pipeline item even
            # when the underlying value is shared. Correctness still depends
            # on ``pipeline_inputs`` (and thus the wrappers) being held by a
            # strong reference for the whole run so the GC cannot recycle an
            # id() while the map is live.
            origin_state: dict = {
                id(ref): {"data_id": did, "error": None, "successes": 0, "item": item}
                for (_, ref), (item, did) in zip(pipeline_inputs, to_process)
            }

            try:
                async for envelope in run_worker_pipeline(
                    tasks=tasks,
                    data_iterable=pipeline_inputs,
                    user=user,
                    ctx=ctx,
                    data_per_batch=data_per_batch,
                    pipeline_name=pipeline_id,
                ):
                    state = origin_state.get(id(envelope.origin))
                    if state is None:
                        # Defensive: origin not in our state map (shouldn't happen).
                        continue
                    if isinstance(envelope.value, _ErroredItem):
                        if state["error"] is None:
                            state["error"] = envelope.value.exception
                            logger.error(
                                f"Item failed in pipeline: {envelope.value.exception}",
                                exc_info=envelope.value.exception,
                            )
                        continue
                    state["successes"] += 1
                    yield PipelineRunYield(
                        pipeline_run_id=pipeline_run_id,
                        dataset_id=dataset.id,
                        dataset_name=dataset.name,
                        payload=envelope.value,
                    )
            except Exception as worker_pipeline_error:
                # An exception that escaped the worker pipeline itself (not a
                # per-item error envelope). Surface it the same way the old
                # gather did: log + raise PipelineRunFailedError.
                logger.error(
                    f"Worker pipeline failed: {worker_pipeline_error}",
                    exc_info=worker_pipeline_error,
                )
                raise PipelineRunFailedError(
                    message="Pipeline run failed. Data item could not be processed."
                ) from worker_pipeline_error

            # Per-item post-flight: mark completed in DB for incremental mode,
            # and build the per-item results dict. Iterate the same _OriginRef
            # wrappers used as origin_state keys so duplicate item objects
            # don't alias to a single state entry.
            for (_, ref), (item, did) in zip(pipeline_inputs, to_process):
                state = origin_state[id(ref)]
                if state["error"] is not None:
                    # Detailed exception is preserved in logs (above) and in
                    # ``item_failures`` on PipelineRunFailedError. The yielded
                    # payload is intentionally sanitized so per-item results
                    # never expose internal paths, upstream payloads, or other
                    # potentially sensitive content from arbitrary exceptions.
                    results.append(
                        {
                            "run_info": PipelineRunErrored(
                                pipeline_run_id=pipeline_run_id,
                                payload="Data item could not be processed.",
                                dataset_id=dataset.id,
                                dataset_name=dataset.name,
                            ),
                            "data_id": did,
                        }
                    )
                else:
                    if incremental_loading and did is not None:
                        await _mark_completed(did, pipeline_name, dataset.id)
                    results.append(
                        {
                            "run_info": PipelineRunCompleted(
                                pipeline_run_id=pipeline_run_id,
                                dataset_id=dataset.id,
                                dataset_name=dataset.name,
                            ),
                            "data_id": did,
                        }
                    )

            item_failures = [
                PipelineItemFailure(data_id=state["data_id"], exception=state["error"])
                for state in origin_state.values()
                if state["error"] is not None
            ]
            if item_failures:
                raise PipelineRunFailedError(
                    message=f"Pipeline run failed: {len(item_failures)} item(s) could not be processed.",
                    item_failures=item_failures,
                )

            await log_pipeline_run_complete(
                pipeline_run_id, pipeline_id, pipeline_name, dataset.id, data
            )

            _safe_send_telemetry(
                "Pipeline Run Completed",
                user.id,
                additional_properties=_pipeline_telemetry_props(pipeline_name, user),
            )

            yield PipelineRunCompleted(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                data_ingestion_info=results,
            )

            graph_engine = await get_graph_engine()
            if hasattr(graph_engine, "push_to_s3"):
                await graph_engine.push_to_s3()

            relational_engine = get_relational_engine()
            if hasattr(relational_engine, "push_to_s3"):
                await relational_engine.push_to_s3()

        except Exception as error:
            await log_pipeline_run_error(
                pipeline_run_id, pipeline_id, pipeline_name, dataset.id, data, error
            )

            _safe_send_telemetry(
                "Pipeline Run Errored",
                user.id,
                additional_properties=_pipeline_telemetry_props(pipeline_name, user),
            )

            yield PipelineRunErrored(
                pipeline_run_id=pipeline_run_id,
                payload=repr(error),
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                data_ingestion_info=locals().get("results"),
            )

            # Mirror previous behavior: surface non-PipelineRunFailedError errors,
            # but absorb PipelineRunFailedError so the caller still sees the
            # yielded Errored info (without an exception bubbling further).
            if not isinstance(error, PipelineRunFailedError):
                raise error
