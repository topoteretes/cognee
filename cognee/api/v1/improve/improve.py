"""``improve()``: the public entry point and orchestrator of the self-improvement loop.

``execute_stages``, defined first inside ``improve()``, is what a run does: one
gated stage after another, in registry order. The body below it decides when
and whether — forward to a remote server, resolve the request, claim the
improve lock, then run in the foreground or detach as one background task.

Two contracts shape everything here. The ``record_operation`` row is the
stage-8 watermark, so it must describe the *finished* run: an errored stage
records ``failed``, a run in which nothing executed records ``noop``, and a
background run defers the row close to the detached task. And the run fails
open: every stage failure is recorded and the next stage still runs — except
the one ``fatal`` stage, ``persist_session_qa``, where losing session Q&A
would be data loss: it stops the run and raises, carrying the partial
``ImproveResult`` on the exception (decision D2).
"""

import asyncio
import hashlib
from collections.abc import Callable, Coroutine, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from typing_extensions import TypedDict, Unpack

from cognee.api.v1.serve.state import get_remote_client
from cognee.infrastructure.background_tasks import register_background_task
from cognee.infrastructure.locks.session_lock import (
    improve_lock_keys,
    release_improve_lock_many,
    try_acquire_improve_lock_many,
)
from cognee.modules.improve import (
    DEFAULT_STAGES,
    MEMIFY_PASSTHROUGH_KEYS,
    REASON_ABORTED_BY_FATAL_STAGE,
    REASON_LOCK_HELD,
    BaseStage,
    ImproveResult,
    ImproveRunInputs,
    StageResult,
    execute_stage,
    get_improve_config,
    resolve_graph_capabilities,
    stage_names,
    validate_stages_disabled,
)
from cognee.modules.migrations.startup import run_migrations_and_block
from cognee.modules.observability import (
    COGNEE_DATASET_NAME,
    COGNEE_IMPROVE_STAGES,
    COGNEE_SESSION_ID,
    new_span,
)
from cognee.modules.operations import finish_operation, record_operation
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger("improve")


class ImproveKwargs(TypedDict, total=False):
    """Power-user overrides for improve(). Most users never need these."""

    extraction_tasks: list
    enrichment_tasks: list
    data: Any
    node_type: type
    user: object
    vector_db_config: dict
    graph_db_config: dict
    feedback_alpha: float


async def improve(
    dataset: str | UUID = "main_dataset",
    *,
    run_in_background: bool = False,
    node_name: list[str] | None = None,
    session_ids: list[str] | None = None,
    build_global_context_index: bool = False,
    build_truth_subspace: bool = False,
    **kwargs: Unpack[ImproveKwargs],
) -> ImproveResult:
    """Run the self-improvement loop over a dataset and report what each stage did.

    The stages are ``cognee.modules.improve.DEFAULT_STAGES``, in order:
    ``feedback_weights``, ``persist_session_qa``, ``persist_agent_traces``,
    ``extract_agent_context``, ``distill_sessions``, ``update_user_preferences``,
    ``build_truth_subspace``, ``triplet_enrichment``, ``global_context_index``.
    Each stage first *gates* — declines work it cannot do under the current
    settings, with zero LLM calls — and only then runs. That registry is the
    authoritative description of what runs.

    Every stage but the last two is session-fed and is skipped with
    ``no_session_ids`` when no ``session_ids`` were given. Stages named in
    ``IMPROVE_STAGES_DISABLED`` are skipped with ``disabled_by_config``. A run
    that loses the improve lock — another run is already touching the same
    sessions or dataset — returns a result whose every stage is
    ``skipped: lock_held``. A failure in ``persist_session_qa`` stops the run
    and raises, because silently losing session Q&A would be data loss; every
    other failure is recorded and the remaining stages still run.

    Args:
        dataset: Dataset name or UUID to process. Resolved once; every stage
            receives the resolved id.
        run_in_background: Run all stages as one background task that holds
            the improve lock for its lifetime. The returned result has
            ``status == "running"``; ``await result.wait()`` blocks on it.
        node_name: Filter graph to specific named entities (enrichment stage).
        session_ids: Session IDs whose feedback and content should be
            bridged into the permanent graph.
        build_global_context_index: Opt in to ``global_context_index``.
        build_truth_subspace: Opt in to ``build_truth_subspace``.
        **kwargs: Additional options — see ``ImproveKwargs``.

    Returns:
        ``ImproveResult`` with one ``StageResult`` per stage, in order. The
        legacy memify run info stays reachable as ``result.memify_run``.

    Example::

        result = await cognee.improve(dataset="docs", session_ids=["chat_1"])
        for stage in result.stages:
            print(stage.stage, stage.status, stage.reason or "")
    """

    async def execute_stages(
        inputs: ImproveRunInputs,
        result: ImproveResult,
        lock_keys: tuple[str, ...],
        operation: Any,
    ) -> None:
        """One ``StageResult`` per registry stage, in order; then free the lock."""
        from cognee.modules.improve.graph_changes import (
            ENRICHMENT_WATERMARK_KEY,
            enrichment_watermark_stamp,
        )

        stages = list(DEFAULT_STAGES)
        try:
            for index, stage in enumerate(stages):
                stage_started_at = datetime.now(timezone.utc)
                stage_result = await execute_stage(stage, inputs)
                result.record(stage_result)
                if (
                    stage.name == ENRICHMENT_WATERMARK_KEY
                    and stage_result.status in ("completed", "already_completed")
                    and not inputs.node_name
                    and not inputs.has_custom_memify_tasks
                ):
                    # The stage-8 watermark: only a full, unscoped enrichment
                    # that actually ran (or verified nothing changed) gates a
                    # later run — a skipped stage 8 must never stamp, and the
                    # stamp carries the stage START so a write racing the row
                    # close stays visible to the next gate.
                    operation.set_run_info(
                        enrichment_watermark_stamp(stage_result.status, stage_started_at)
                    )
                if stage.fatal and stage_result.status == "errored":
                    raise _abort_run(result, stages[index + 1 :], stage, stage_result)
        finally:
            result.finished = True
            from cognee.modules.pipelines.models import OperationOutcome

            if result.status == "errored":
                # A non-fatal errored stage exits this block cleanly; without
                # this the row would say "succeeded" and gate off the retry.
                operation.set_outcome(OperationOutcome.FAILED)
            elif result.status == "skipped":
                # Every stage skipped means nothing ran, so there is nothing
                # to watermark: a "succeeded" row would gate enrichment off
                # until the dataset's next write pipeline.
                operation.set_outcome(OperationOutcome.NOOP)
            await release_improve_lock_many(lock_keys)

    session_ids = [session_id for session_id in (session_ids or []) if session_id]
    _send_improve_telemetry(
        dataset,
        session_ids,
        user=kwargs.get("user", "sdk"),
        run_in_background=run_in_background,
        build_global_context_index=build_global_context_index,
        build_truth_subspace=build_truth_subspace,
    )

    with _improve_span(dataset, session_ids) as report:
        remote_client = get_remote_client()
        if remote_client is not None:
            remote_result = await _improve_remotely(
                remote_client,
                dataset,
                node_name=node_name,
                session_ids=session_ids,
                build_global_context_index=build_global_context_index,
                build_truth_subspace=build_truth_subspace,
                run_in_background=run_in_background,
                overrides=kwargs,
            )
            return report(remote_result)

        # Opened across the stage execution, not just the prep: this row is
        # the stage-8 watermark and must describe the finished run, never the
        # launch (outcome contract in the module docstring).
        async with record_operation("improve") as operation:
            inputs = await _resolve_inputs(
                operation,
                dataset=dataset,
                session_ids=session_ids,
                run_in_background=run_in_background,
                node_name=node_name,
                build_global_context_index=build_global_context_index,
                build_truth_subspace=build_truth_subspace,
                overrides=kwargs,
            )
            # One claim per session id plus one for the dataset, so bridge
            # runs and dataset runs exclude each other; held until the last
            # stage finishes, background included.
            lock_keys = improve_lock_keys(inputs.session_ids, inputs.dataset_id, inputs.user.id)
            if not await try_acquire_improve_lock_many(lock_keys):
                return report(_skip_lock_held_run(operation, inputs, lock_keys))

            # Created before the stages run: background mode hands this result
            # to the caller while the detached task is still filling it.
            result = ImproveResult(
                dataset_id=inputs.dataset_id,
                dataset_name=inputs.dataset_name,
                session_ids=inputs.session_id_list,
                memify_run={},
                background=run_in_background,
                finished=False,
            )

            if run_in_background:
                operation.defer_close()
                # ``operation`` is passed twice on purpose: the coroutine's own
                # finally sets the outcome from the finished stages, while
                # _run_detached closes the deferred row even on cancellation.
                run = _run_detached(execute_stages(inputs, result, lock_keys, operation), operation)
                task = register_background_task(asyncio.create_task(run))
                result.attach_background_task(task)
                return report(result)

            await execute_stages(inputs, result, lock_keys, operation)
            return report(result)


@contextmanager
def _improve_span(
    dataset: str | UUID, session_ids: list[str]
) -> Iterator[Callable[[ImproveResult], ImproveResult]]:
    """The tracing span around one improve call.

    Yields ``report``: call it with the result on the way out to stamp the
    stage summary onto the span — ``"background"`` for a detached run, whose
    stages land only after the span is gone. A fatal stage raises instead of
    returning; the partial result on the exception stamps how far it got.
    """
    with new_span("cognee.api.improve") as span:
        span.set_attribute(COGNEE_DATASET_NAME, str(dataset))
        if session_ids:
            span.set_attribute(COGNEE_SESSION_ID, ",".join(session_ids))

        def report(result: ImproveResult) -> ImproveResult:
            span.set_attribute(
                COGNEE_IMPROVE_STAGES,
                "background" if result.status == "running" else result.stage_summary(),
            )
            return result

        try:
            yield report
        except Exception as error:
            partial_result = getattr(error, "improve_result", None)
            if partial_result is not None:
                span.set_attribute(COGNEE_IMPROVE_STAGES, partial_result.stage_summary())
            raise


async def _resolve_inputs(
    operation: Any,
    *,
    dataset: str | UUID,
    session_ids: list[str],
    run_in_background: bool,
    node_name: list[str] | None,
    build_global_context_index: bool,
    build_truth_subspace: bool,
    overrides: dict,
) -> ImproveRunInputs:
    """Resolve everything a stage may read into the frozen ``ImproveRunInputs``.

    ``operation`` is the open ``record_operation`` context; each fact is bound
    to it as soon as it is known, so a run that fails to resolve its dataset is
    still recorded as a failed improve. The options arrive one by one on
    purpose: a bundle type would be a third copy of ``improve()``'s signature
    to keep in sync. ``overrides`` is the caller's ``**kwargs``, never mutated.
    """
    user = overrides.get("user")
    if user is None:
        user = await get_default_user()
    operation.set_user(user)
    operation.set_background(run_in_background)
    if len(session_ids) == 1:
        operation.set_session_id(session_ids[0])

    # The run-log writers INSERT the operation-record columns, so the database
    # must be at the current Alembic head before the first write — same gate
    # as cognify().
    await run_migrations_and_block(dataset, user)

    # The same write-level resolver remember/memify use: names resolve or are
    # created for the caller; a missing or unauthorized UUID raises instead of
    # being silently retargeted. Downstream gets the resolved UUID, never a
    # name — names are owner-scoped, so a name collapsed from a *shared*
    # dataset's UUID would re-resolve to the caller's own same-named dataset.
    user, authorized_datasets = await resolve_authorized_user_datasets(dataset, user)
    resolved_dataset = authorized_datasets[0]
    operation.set_dataset(resolved_dataset.id)

    config = get_improve_config()
    # Fail loudly rather than skip silently in the loop: a typo in
    # IMPROVE_STAGES_DISABLED would disable nothing, and the fatal stage must
    # not be bypassable by config.
    validate_stages_disabled(config.stages_disabled, DEFAULT_STAGES)
    feedback_alpha = overrides.get("feedback_alpha")
    if feedback_alpha is None:
        feedback_alpha = config.feedback_alpha

    return ImproveRunInputs(
        user=user,
        dataset_id=resolved_dataset.id,
        dataset=resolved_dataset,
        improve_operation_id=operation.operation_id,
        session_ids=tuple(session_ids),
        config=config,
        capabilities=await resolve_graph_capabilities(
            resolved_dataset.id, getattr(resolved_dataset, "owner_id", None)
        ),
        node_name=node_name,
        feedback_alpha=feedback_alpha,
        build_global_context_index=build_global_context_index,
        build_truth_subspace=build_truth_subspace,
        memify_kwargs={key: overrides[key] for key in MEMIFY_PASSTHROUGH_KEYS if key in overrides},
    )


def _skip_lock_held_run(
    operation: Any, inputs: ImproveRunInputs, lock_keys: tuple[str, ...]
) -> ImproveResult:
    """React to a lost lock claim: log it, record a no-op run, skip every stage.

    Not "succeeded": zero stages ran, and the dataset is already bound to the
    record, so a succeeded row would stand in as the stage-8 watermark for a
    dataset this claim may never have improved (a clash on a shared session
    key). The caller still gets one entry per stage, never ``{}``.
    """
    from cognee.modules.pipelines.models import OperationOutcome

    logger.info(
        "improve: another run holds the improve lock for %s, skipping",
        ", ".join(lock_keys),
    )
    operation.set_outcome(OperationOutcome.NOOP)
    return ImproveResult.all_skipped(
        stage_names(DEFAULT_STAGES),
        REASON_LOCK_HELD,
        dataset_id=inputs.dataset_id,
        dataset_name=inputs.dataset_name,
        session_ids=inputs.session_id_list,
    )


def _abort_run(
    result: ImproveResult,
    remaining_stages: Sequence[BaseStage],
    stage: BaseStage,
    stage_result: StageResult,
) -> BaseException:
    """Mark the rest of the run aborted and build the exception the loop raises.

    The exception is the stage's own when it raised — its type is what the HTTP
    layer maps onto a status code — and a synthetic one when the wrapped
    pipeline only reported ``PipelineRunErrored`` without raising. Either way it
    carries the partial ``ImproveResult`` as ``improve_result``.
    """
    for remaining_stage in remaining_stages:
        result.record(StageResult.skipped(remaining_stage.name, REASON_ABORTED_BY_FATAL_STAGE))
    result.error = stage_result.error
    logger.error(
        "improve: fatal stage '%s' failed, run stopped: %s", stage.name, stage_result.error
    )

    # Lazy: cognee.exceptions pulls in fastapi, and this path runs only when a
    # fatal stage has already failed.
    from cognee.exceptions import CogneeSystemError

    error = stage_result.exception
    if error is None:
        error = CogneeSystemError(
            message=f"improve: fatal stage '{stage.name}' errored: {stage_result.error}",
            name="ImproveFatalStageError",
            log=False,
        )

    try:
        error.improve_result = result  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Exceptions that reject attribute assignment (__slots__, some C-level
        # types) simply carry no partial result.
        logger.debug("improve: could not attach partial result to %s", type(error).__name__)

    return error


async def _run_detached(execute: Coroutine[Any, Any, None], operation: Any) -> None:
    """Await the stage execution where a fatal stage has nowhere to raise.

    The failure is logged and already on ``result.error`` for whoever awaits
    ``result.wait()``. The deferred row close in ``finally`` is unconditional:
    even a cancelled run records ``failed: CancelledError`` — with the close
    deferred, skipping it would leave the run with no row at all.
    """
    error: BaseException | None = None
    try:
        await execute
    except Exception as caught:
        error = caught
        logger.warning("improve: background run aborted by fatal stage: %s", caught, exc_info=True)
    except BaseException as caught:
        error = caught
        raise
    finally:
        await finish_operation(operation, error=error)


async def _improve_remotely(
    remote_client: Any,
    dataset: str | UUID,
    *,
    node_name: list[str] | None,
    session_ids: list[str],
    build_global_context_index: bool,
    build_truth_subspace: bool,
    run_in_background: bool,
    overrides: dict,
) -> ImproveResult:
    """Forward every option to the configured server, which runs the same
    stages and hands back its ``ImproveResult``.

    ``run_in_background=True`` over serve() is fire-and-forget: the server
    returns the running result and finishes on its own, but no local task is
    attached, so ``result.status`` stays ``"running"`` and ``await
    result.wait()`` returns immediately. There is no polling endpoint yet.
    """
    payload = await remote_client.improve(
        dataset,
        node_name=node_name,
        session_ids=session_ids or None,
        build_global_context_index=build_global_context_index,
        build_truth_subspace=build_truth_subspace,
        run_in_background=run_in_background,
        **overrides,
    )
    return ImproveResult.from_remote_payload(payload, session_ids)


def _send_improve_telemetry(
    dataset: str | UUID,
    session_ids: list[str],
    *,
    user: Any,
    run_in_background: bool,
    build_global_context_index: bool,
    build_truth_subspace: bool,
) -> None:
    # cognee/__init__.py imports this module, so the version has to be read at
    # call time; its own NOTE explains why __version__ sits at the top there.
    from cognee import __version__ as cognee_version

    send_telemetry(
        "cognee.improve",
        user,
        additional_properties={
            "dataset": str(dataset),
            "session_count": len(session_ids),
            # Hashed, never raw: session ids are user-chosen strings.
            "session_ids": ",".join(_hash_session_id(sid) for sid in session_ids),
            "run_in_background": run_in_background,
            "build_global_context_index": build_global_context_index,
            "build_truth_subspace": build_truth_subspace,
            "cognee_version": cognee_version,
        },
    )


def _hash_session_id(session_id: str) -> str:
    """Short, stable, non-reversible token for telemetry — never the raw id."""
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16]
