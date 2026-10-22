"""Durable single-row records for non-pipeline cognee operations (SDK-399).

``record_operation("search")`` wraps an operation body and, on exit, writes
exactly one ``pipeline_runs`` row with ``status = NULL`` carrying the
operation name, triggering user/tenant, start/end timestamps, outcome
("succeeded"/"failed"), the error class on failure, and the LLM tokens
spent inside the scope. NULL-status rows are invisible to all legacy
latest-row status readers (they filter on ``pipeline_name``/``status``).

An operation that starts background work outliving the scope can defer the
write (``context.defer_close()``) and have that work close the record with
``finish_operation``, so the row describes the finished work, not the launch.

Guarantees:
- The wrapped operation's exceptions always propagate unchanged.
- The recorder's own persistence failures are logged and swallowed — it
  can never break the operation it records. (Known consequence: the
  ``prune_system(metadata=True)`` record is self-erasing — it drops the
  relational DB including ``pipeline_runs``, so its own write fails and
  is swallowed by design.)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlalchemy.exc import OperationalError

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .origin import get_operation_origin
from .scrub_error import scrub_error_message
from .usage_accumulator import (
    OperationUsage,
    get_parent_run_id,
    operation_usage_scope,
    parent_run_scope,
)

if TYPE_CHECKING:
    from cognee.modules.users.models import User

logger = get_logger("record_operation")


_current_operation: ContextVar[Optional["OperationContext"]] = ContextVar(
    "cognee_current_operation", default=None
)


def get_current_operation() -> Optional["OperationContext"]:
    """Return the innermost active operation context, if any.

    Deep call sites that resolve the user lazily (e.g. recall's graph
    path) use this to bind attribution without signature plumbing.
    """
    return _current_operation.get()


class OperationContext:
    """Mutable attribution context for one recorded operation."""

    def __init__(
        self,
        operation_name: str,
        user_id: UUID | None = None,
        tenant_id: UUID | None = None,
        dataset_id: UUID | None = None,
        usage: OperationUsage | None = None,
        session_id: str | None = None,
        background: bool | None = None,
        parent_operation_id: UUID | None = None,
    ):
        self.operation_name = operation_name
        # Allocated up front (not at write time) so children — nested
        # record_operation scopes and pipeline log writers — can reference
        # this operation as parent_operation_id before its row exists.
        # Persisted as the row's pipeline_run_id.
        self.operation_id: UUID = uuid4()
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.dataset_id = dataset_id
        self.usage = usage or OperationUsage()
        self.session_id = session_id
        self.background = background
        self.parent_operation_id = parent_operation_id
        # The eval-capture run scope opened for this operation, when capture is
        # active (SDK-529). Bound by ``record_operation`` so ``set_dataset``
        # can forward the dataset the moment the caller learns it.
        self._capture_scope = None
        self.started_at = datetime.now(timezone.utc)
        # An explicit outcome for the row, set by operations whose success is
        # not "the body did not raise" (improve marks a run with an errored
        # stage as failed). A raised exception still wins over it.
        self.outcome = None
        self.close_deferred = False
        # Optional JSON payload for the row's run_info column (improve stamps
        # its enrichment watermark here — see improve/graph_changes.py).
        self.run_info = None

    def set_user(self, user) -> None:
        """Bind the triggering user (tolerates None and partial objects).

        Forwarded to the eval-capture scope like ``set_dataset``, so records
        buffered before the caller knew its user still carry it.
        """
        if user is None:
            return
        self.user_id = getattr(user, "id", None)
        self.tenant_id = getattr(user, "tenant_id", None)
        capture_scope = self._capture_scope
        if capture_scope is not None:
            capture_scope.set_user(self.user_id, self.tenant_id)

    def set_dataset(self, dataset_id: UUID | None) -> None:
        """Bind the target dataset, when the operation has exactly one.

        Forwarded to the eval-capture scope immediately, not at operation exit:
        capture resolves an event's dataset when the event is FLUSHED, and the
        flusher ticks every ``FLUSH_INTERVAL_S`` (2 s) while a search's LLM
        completion runs for longer than that. Binding only in the ``finally``
        filed essentially every search's ``retrieval.candidates`` under
        ``nodataset/`` while its manifest landed under ``<dataset>/``.
        """
        self.dataset_id = dataset_id
        capture_scope = self._capture_scope
        if capture_scope is not None:
            capture_scope.set_dataset(dataset_id)

    def set_session_id(self, session_id: str | None) -> None:
        """Bind the active session-cache id (joins SessionModelUsage)."""
        if session_id:
            self.session_id = session_id

    def set_background(self, background: bool) -> None:
        """Mark whether this call launched background work.

        True means outcome="succeeded" records "accepted and started", not
        "background work finished" (SDK-399 background-launch semantics) —
        unless the operation defers its close to that work (``defer_close``),
        which improve does so its row describes the finished run.
        """
        self.background = background

    def set_outcome(self, outcome) -> None:
        """Override the outcome recorded for a body that exits cleanly."""
        self.outcome = outcome

    def merge_run_info(self, run_info: dict, *, allow_overwrite: bool = False) -> None:
        """Merge a payload into the row's ``run_info`` column, append-style.

        Writers namespace their entries by key (improve stages stamp under
        their stage name), so a merge never drops an earlier writer's entry.
        Overwriting an existing key is legal (last writer wins) but logged:
        it means two writers chose the same namespace — unless the caller
        says the overwrite is intended (``allow_overwrite``: an improve rerun
        pass refreshing its own stage's stamp).
        """
        existing = self.run_info or {}
        clobbered = [key for key in run_info if key in existing and existing[key] != run_info[key]]
        if clobbered and not allow_overwrite:
            logger.warning(
                "record_operation: run_info keys overwritten on %s record: %s",
                self.operation_name,
                ", ".join(sorted(clobbered)),
            )
        self.run_info = {**existing, **run_info}

    def defer_close(self) -> None:
        """Hand the row write to background work — see ``finish_operation``.

        The ``record_operation`` scope then writes nothing on exit; the work
        that outlives it must call ``finish_operation`` exactly once.
        """
        self.close_deferred = True


async def _write_operation_row(
    context: OperationContext,
    started_at: datetime,
    outcome: str,
    error_class: str | None,
    error_message: str | None,
) -> None:
    from cognee.modules.pipelines.models import PipelineRun

    pipeline_run = PipelineRun(
        status=None,
        pipeline_run_id=context.operation_id,
        pipeline_name=None,
        pipeline_id=None,
        dataset_id=context.dataset_id,
        run_info=context.run_info,
        user_id=context.user_id,
        tenant_id=context.tenant_id,
        operation_name=context.operation_name,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        outcome=outcome,
        error_class=error_class,
        error_message=error_message,
        tokens_in=context.usage.tokens_in,
        tokens_out=context.usage.tokens_out,
        embedding_tokens=context.usage.embedding_tokens,
        origin=get_operation_origin(),
        session_id=context.session_id,
        parent_operation_id=context.parent_operation_id,
        background=context.background,
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()


@asynccontextmanager
async def record_operation(
    operation_name: str,
    user: Optional["User"] = None,
    dataset_id: UUID | None = None,
    session_id: str | None = None,
    background: bool | None = None,
) -> AsyncIterator[OperationContext]:
    """Record one non-pipeline operation as a single ``pipeline_runs`` row."""
    from cognee.modules.pipelines.models import OperationOutcome

    with operation_usage_scope() as usage:
        context = OperationContext(
            operation_name=operation_name,
            dataset_id=dataset_id,
            usage=usage,
            session_id=session_id,
            background=background,
            # The innermost enclosing run — an operation OR a pipeline (a
            # search inside a custom pipeline parents to that pipeline run).
            parent_operation_id=get_parent_run_id(),
        )
        if user is not None:
            context.set_user(user)

        context_token = _current_operation.set(context)

        outcome = OperationOutcome.SUCCEEDED
        error_class: str | None = None
        error_message: str | None = None

        # Eval capture (SDK-529). Imported lazily so ``import cognee`` never pays
        # for the capture package; is_active() is one global read once
        # initialized, and the OFF path constructs no scope at all. No drain()
        # here: this wraps the user-facing recall/search path, and the manifest
        # lands within FLUSH_INTERVAL_S like any other event. The manifest
        # mirrors the row's attribution fields (operation, outcome,
        # error_class) so a captured run can be filtered offline without
        # joining back to ``pipeline_runs``.
        from cognee.modules.observability import capture as eval_capture

        capture_scope_cm = (
            eval_capture.run_scope(
                context.operation_id,
                dataset_id,
                kind="operation",
                user_id=context.user_id,
                tenant_id=context.tenant_id,
            )
            if eval_capture.is_active()
            else nullcontext()
        )
        with capture_scope_cm as capture_scope:
            if capture_scope is not None:
                capture_scope.note("operation", operation_name)
                # So context.set_dataset() reaches the scope as soon as the
                # caller resolves its dataset; the ``finally`` below is only a
                # backstop for callers that never call it.
                context._capture_scope = capture_scope
            try:
                with parent_run_scope(context.operation_id):
                    yield context
            except BaseException as exc:
                outcome = OperationOutcome.FAILED
                error_class = type(exc).__name__
                error_message = scrub_error_message(exc)
                raise
            finally:
                _current_operation.reset(context_token)
                # An operation may declare its own outcome (improve marks a run
                # with an errored stage as failed); a raised exception still wins.
                if outcome is OperationOutcome.SUCCEEDED and context.outcome is not None:
                    outcome = context.outcome
                if capture_scope is not None:
                    # Backstop: set_dataset() already forwarded the dataset when
                    # the caller bound it; this covers a dataset passed at entry.
                    capture_scope.set_dataset(context.dataset_id)
                    capture_scope.note("outcome", outcome.value)
                    # None on success, kept for a stable manifest shape.
                    capture_scope.note("error_class", error_class)
                    # A deferred close means the row — with the real end time and
                    # outcome — is written later by finish_operation(); this
                    # manifest describes the launch.
                    capture_scope.note("close_deferred", context.close_deferred)
                # A deferred close writes nothing here: the background work the
                # operation started owns the row (``finish_operation``), so it
                # carries the run's end time and outcome, not the launch's.
                if not context.close_deferred:
                    try:
                        await _write_operation_row(
                            context, context.started_at, outcome.value, error_class, error_message
                        )
                    except OperationalError as write_error:
                        # An unreachable relational store is an expected state
                        # around the operations that manage the store itself:
                        # prune deletes the database the ledger lives in, and
                        # nothing exists before setup() — the quickstart examples
                        # hit both. A warning with a traceback made every example
                        # run look broken, so the skipped write logs at debug.
                        logger.debug(
                            "record_operation: skipping %s record, relational store unavailable (%s)",
                            operation_name,
                            write_error,
                        )
                    except Exception as write_error:
                        logger.warning(
                            "record_operation: failed to persist %s record (%s)",
                            operation_name,
                            write_error,
                            exc_info=True,
                        )


async def finish_operation(context: OperationContext, error: BaseException | None = None) -> None:
    """Write the row for an operation that deferred its close (``defer_close``).

    Called by the background work that outlives the ``record_operation``
    scope, so the row's ``ended_at`` and outcome describe the finished work,
    not the launch. Swallows its own persistence failures, like the scope.
    """
    from cognee.modules.pipelines.models import OperationOutcome

    if error is not None:
        outcome = OperationOutcome.FAILED
        error_class: str | None = type(error).__name__
        error_message: str | None = scrub_error_message(error)
    else:
        outcome = context.outcome or OperationOutcome.SUCCEEDED
        error_class = None
        error_message = None

    try:
        await _write_operation_row(
            context, context.started_at, outcome.value, error_class, error_message
        )
    except OperationalError as write_error:
        # Store unavailable: expected around prune / before setup(), see the
        # matching handler in record_operation.
        logger.debug(
            "record_operation: skipping %s record, relational store unavailable (%s)",
            context.operation_name,
            write_error,
        )
    except Exception as write_error:
        logger.warning(
            "record_operation: failed to persist %s record (%s)",
            context.operation_name,
            write_error,
            exc_info=True,
        )
