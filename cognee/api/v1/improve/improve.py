"""``improve()``: the orchestrator of the self-improvement loop.

It resolves the dataset once, claims one lock keyed to the run, assembles a
frozen ``ImproveRunInputs``, and walks ``DEFAULT_STAGES`` in registry order:
gate, then run, timing each stage and mapping its outcome onto a
``StageResult``. The stage bodies live in ``cognee/modules/improve/stages.py``
and the code they wrap stays where it always was; this module owns only the
glue (plan Part 5.1).
"""

import asyncio
import hashlib
import time
from typing import Any, List, Optional, Type, Union
from uuid import UUID

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.modules.improve import (
    DEFAULT_STAGES,
    MEMIFY_PASSTHROUGH_KEYS,
    REASON_ABORTED_BY_FATAL_STAGE,
    REASON_LOCK_HELD,
    ImproveResult,
    ImproveRunInputs,
    StageResult,
    evaluate_gate,
    get_improve_config,
    resolve_graph_capabilities,
    stage_names,
)
from cognee.modules.observability import (
    COGNEE_DATASET_NAME,
    COGNEE_IMPROVE_STAGES,
    COGNEE_SESSION_ID,
    new_span,
)
from cognee.modules.operations import record_operation
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("improve")

# Strong refs for background improve chains. The event loop only keeps weak
# references to tasks, so without anchoring here gc can collect an in-flight
# chain mid-run (same pattern as remember.py's _BACKGROUND_REMEMBER_TASKS).
_BACKGROUND_IMPROVE_TASKS: set = set()


class ImproveKwargs(TypedDict, total=False):
    """Power-user overrides for improve(). Most users never need these."""

    extraction_tasks: list
    enrichment_tasks: list
    data: Any
    node_type: Type
    user: object
    vector_db_config: dict
    graph_db_config: dict
    feedback_alpha: float


def _hash_session_id(session_id: str) -> str:
    """Short, stable, non-reversible token for telemetry — never the raw id."""
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16]


async def improve(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    run_in_background: bool = False,
    node_name: Optional[List[str]] = None,
    session_ids: Optional[List[str]] = None,
    build_global_context_index: bool = False,
    build_truth_subspace: bool = False,
    **kwargs: Unpack[ImproveKwargs],
) -> ImproveResult:
    """Run the self-improvement loop over a dataset and return what each stage did.

    The nine stages run in a fixed order (``cognee.modules.improve.DEFAULT_STAGES``).
    Each one first *gates* — declines work it cannot do under the current
    settings, with zero LLM calls — and only then runs:

    1. ``feedback_weights`` — scored session answers move ``feedback_weight`` on
       the graph elements they used. Skipped at ``DEFAULT_FEEDBACK_INFLUENCE=0``
       (``feedback_influence_zero``) or on backends without the method
       (``backend_unsupported``).
    2. ``persist_session_qa`` — session Q&A is cognified into the graph
       (``user_sessions_from_cache``). The one fatal stage: an error stops the
       chain and is raised, because silently losing Q&A would be data loss.
    3. ``persist_agent_traces`` — tool-call trace feedback is cognified
       (``agent_trace_feedbacks``).
    4. ``extract_agent_context`` — pending traces become agent-profile lessons.
       Skipped when the session cache or ``AUTO_FEEDBACK`` is off.
    5. ``distill_sessions`` — gated guidance becomes entity-anchored lessons
       (``session_learnings``).
    6. ``update_user_preferences`` — ratings fold into ``prefers`` weights.
       Skipped unless ``PERSONALIZATION_ENABLED``.
    7. ``build_truth_subspace`` — opt-in (``build_truth_subspace=True``); also
       needs a backend with truth state.
    8. ``triplet_enrichment`` — default memify enrichment. Skipped when
       ``triplet_embedding`` is off (``triplet_embedding_disabled``);
       ``already_completed`` when no write pipeline finished for this dataset
       since its last improve.
    9. ``global_context_index`` — opt-in (``build_global_context_index=True``).

    Session-kind stages (1-7) are skipped with ``no_session_ids`` when no
    ``session_ids`` were given. Stages named in ``IMPROVE_STAGES_DISABLED`` are
    skipped with ``disabled_by_config``. A run that loses the improve lock
    (another run is already touching the same sessions or dataset) returns a
    result whose every stage is ``skipped: lock_held``.

    Args:
        dataset: Dataset name or UUID to process. Resolved once; every stage
            receives the resolved id.
        run_in_background: Run the whole chain as one background task that
            holds the improve lock for its lifetime. The returned result has
            ``status == "running"``; ``await result.wait()`` blocks on it.
        node_name: Filter graph to specific named entities (enrichment stage).
        session_ids: Session IDs whose feedback and content should be
            bridged into the permanent graph.
        build_global_context_index: Opt in to stage 9.
        build_truth_subspace: Opt in to stage 7.
        **kwargs: Additional options — see ``ImproveKwargs``.

    Returns:
        ``ImproveResult`` with one ``StageResult`` per stage, in order. The
        legacy memify run info stays reachable as ``result.memify_run``.

    Example::

        result = await cognee.improve(dataset="docs", session_ids=["chat_1"])
        for stage in result.stages:
            print(stage.stage, stage.status, stage.reason or "")
    """
    from cognee import __version__ as cognee_version
    from cognee.shared.utils import send_telemetry

    session_ids = [sid for sid in (session_ids or []) if sid]

    send_telemetry(
        "cognee.improve",
        kwargs.get("user", "sdk"),
        additional_properties={
            "dataset": str(dataset),
            "session_count": len(session_ids),
            # Hashed, never raw: session ids are user-chosen strings (A6).
            "session_ids": ",".join(_hash_session_id(sid) for sid in session_ids),
            "run_in_background": run_in_background,
            "build_global_context_index": build_global_context_index,
            "build_truth_subspace": build_truth_subspace,
            "cognee_version": cognee_version,
        },
    )

    with new_span("cognee.api.improve") as span:
        span.set_attribute(COGNEE_DATASET_NAME, str(dataset))
        if session_ids:
            span.set_attribute(COGNEE_SESSION_ID, ",".join(session_ids))

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()
        if client is not None:
            # Remote mode forwards every option (PR #3824 revived): the server
            # runs the same orchestrator and hands back its ImproveResult.
            payload = await client.improve(
                dataset,
                node_name=node_name,
                session_ids=session_ids or None,
                build_global_context_index=build_global_context_index,
                build_truth_subspace=build_truth_subspace,
                run_in_background=run_in_background,
                **kwargs,
            )
            return _coerce_remote_result(payload, session_ids)

        from cognee.modules.users.methods import get_default_user

        async with record_operation("improve") as operation_context:
            user = kwargs.pop("user", None)
            if user is None:
                user = await get_default_user()
            operation_context.set_user(user)

            # The pipeline-run log writers INSERT the operation-record columns
            # (user_id, outcome, tokens, ...), so an existing database must be
            # at the current Alembic head before the first write — same gate
            # as cognify().
            from cognee.modules.migrations.startup import run_migrations_and_block

            await run_migrations_and_block(dataset, user)

            # One write-level resolution, shared by every stage — the same
            # resolver remember/memify use: names resolve or are created for
            # the caller; a missing or unauthorized UUID raises
            # DatasetNotFoundError instead of being silently retargeted.
            # Downstream always receives the resolved UUID, never a name:
            # names are owner-scoped, so a name collapsed from a *shared*
            # dataset's UUID would re-resolve to the caller's own same-named
            # dataset inside the pipelines.
            user, authorized_datasets = await resolve_authorized_user_datasets(dataset, user)
            resolved_dataset = authorized_datasets[0]
            dataset_id: UUID = resolved_dataset.id
            dataset_name = getattr(resolved_dataset, "name", None)
            operation_context.set_dataset(dataset_id)
            if len(session_ids) == 1:
                operation_context.set_session_id(session_ids[0])
            operation_context.set_background(run_in_background)

            config = get_improve_config()
            feedback_alpha = kwargs.pop("feedback_alpha", None)
            if feedback_alpha is None:
                feedback_alpha = config.feedback_alpha

            capabilities = await resolve_graph_capabilities(
                dataset_id, getattr(resolved_dataset, "owner_id", None)
            )

            inputs = ImproveRunInputs(
                user=user,
                dataset_id=dataset_id,
                dataset=resolved_dataset,
                session_ids=tuple(session_ids),
                config=config,
                capabilities=capabilities,
                node_name=node_name,
                feedback_alpha=feedback_alpha,
                build_global_context_index=build_global_context_index,
                build_truth_subspace=build_truth_subspace,
                memify_kwargs={
                    key: kwargs[key] for key in MEMIFY_PASSTHROUGH_KEYS if key in kwargs
                },
            )

            # One claim per run, keyed by what the run touches: every session
            # id given, or the dataset id when none. Held across the whole
            # chain, background included, so no stage reads while another run
            # is still writing (plan Part 5.8).
            from cognee.infrastructure.locks.session_lock import (
                improve_lock_keys,
                release_improve_lock_many,
                try_acquire_improve_lock_many,
            )

            lock_keys = improve_lock_keys(session_ids, dataset_id)
            if not await try_acquire_improve_lock_many(lock_keys):
                logger.info(
                    "improve: another run holds the improve lock for %s, skipping",
                    ", ".join(lock_keys),
                )
                result = ImproveResult.all_skipped(
                    stage_names(DEFAULT_STAGES),
                    REASON_LOCK_HELD,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    session_ids=session_ids,
                )
                span.set_attribute(COGNEE_IMPROVE_STAGES, result.stage_summary())
                return result

            result = ImproveResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                session_ids=list(session_ids),
                memify_run={},
                background=run_in_background,
                finished=False,
            )

            if run_in_background:

                async def _run_chain_in_background():
                    try:
                        await _run_stages(inputs, result)
                    except Exception as exc:
                        logger.warning("improve: background chain aborted by fatal stage: %s", exc)
                    finally:
                        result.finished = True
                        await release_improve_lock_many(lock_keys)

                task = asyncio.create_task(_run_chain_in_background())
                _BACKGROUND_IMPROVE_TASKS.add(task)
                task.add_done_callback(_BACKGROUND_IMPROVE_TASKS.discard)
                result._task = task
                span.set_attribute(COGNEE_IMPROVE_STAGES, "background")
                return result

            try:
                await _run_stages(inputs, result)
            finally:
                result.finished = True
                await release_improve_lock_many(lock_keys)
                span.set_attribute(COGNEE_IMPROVE_STAGES, result.stage_summary())

            return result


async def _run_stages(inputs: ImproveRunInputs, result: ImproveResult) -> None:
    """Walk the registry: gate, run, time, map. Fills ``result.stages`` in order.

    Every stage's error is recorded as ``errored`` and the chain continues,
    except a ``fatal`` stage (decision D2): the remaining stages are marked
    ``skipped: aborted_by_fatal_stage``, the partial result is attached to
    the exception as ``improve_result`` and the exception is re-raised.
    """
    stages = list(DEFAULT_STAGES)
    for index, stage in enumerate(stages):
        started = time.perf_counter()
        try:
            reason = evaluate_gate(stage, inputs)
        except Exception as exc:
            logger.warning("improve: gate for stage '%s' failed: %s", stage.name, exc)
            reason = None

        if reason is not None:
            result.stages.append(StageResult.skipped(stage.name, reason))
            logger.debug("improve: stage '%s' skipped (%s)", stage.name, reason)
            continue

        try:
            stage_result = await stage.run(inputs)
        except Exception as exc:
            stage_result = StageResult.errored(stage.name, exc)
            stage_result.duration_ms = int((time.perf_counter() - started) * 1000)
            result.stages.append(stage_result)
            if stage.fatal:
                for remaining in stages[index + 1 :]:
                    result.stages.append(
                        StageResult.skipped(remaining.name, REASON_ABORTED_BY_FATAL_STAGE)
                    )
                result.error = stage_result.error
                logger.error("improve: fatal stage '%s' failed, chain stopped: %s", stage.name, exc)
                try:
                    exc.improve_result = result  # type: ignore[attr-defined]
                except Exception:
                    pass
                raise
            logger.warning("improve: stage '%s' failed (non-fatal): %s", stage.name, exc)
            continue

        stage_result.duration_ms = int((time.perf_counter() - started) * 1000)
        result.stages.append(stage_result)
        if stage.name == "triplet_enrichment":
            result.memify_run = stage_result.raw_run if stage_result.raw_run is not None else {}

        if stage.fatal and stage_result.status == "errored":
            # The wrapped pipeline reported PipelineRunErrored instead of
            # raising. Fail closed all the same (D2): stop the chain and raise
            # so the caller sees it — later stages must not report success
            # over lost Q&A.
            from cognee.exceptions import CogneeSystemError

            for remaining in stages[index + 1 :]:
                result.stages.append(
                    StageResult.skipped(remaining.name, REASON_ABORTED_BY_FATAL_STAGE)
                )
            result.error = stage_result.error
            error = CogneeSystemError(
                message=f"improve: fatal stage '{stage.name}' errored: {stage_result.error}",
                name="ImproveFatalStageError",
                log=False,
            )
            error.improve_result = result  # type: ignore[attr-defined]
            raise error


def _coerce_remote_result(payload: Any, session_ids: List[str]) -> ImproveResult:
    """Turn the remote server's JSON into an ``ImproveResult``.

    A server running this orchestrator returns the serialized result; an older
    server returns the legacy memify run mapping, which is nested as
    ``memify_run`` with no stage detail.
    """
    if isinstance(payload, ImproveResult):
        return payload
    if isinstance(payload, dict) and "stages" in payload:
        try:
            payload = {key: value for key, value in payload.items() if key != "status"}
            return ImproveResult.model_validate(payload)
        except Exception as error:
            logger.debug("improve: remote result did not validate as ImproveResult: %s", error)
    return ImproveResult(session_ids=list(session_ids), stages=[], memify_run=payload)
