"""Durable single-row records for non-pipeline cognee operations (SDK-399).

``record_operation("search")`` wraps an operation body and, on exit, writes
exactly one ``pipeline_runs`` row with ``status = NULL`` carrying the
operation name, triggering user/tenant, start/end timestamps, outcome
("succeeded"/"failed"), the error class on failure, and the LLM tokens
spent inside the scope. NULL-status rows are invisible to all legacy
latest-row status readers (they filter on ``pipeline_name``/``status``).

Guarantees:
- The wrapped operation's exceptions always propagate unchanged.
- The recorder's own persistence failures are logged and swallowed — it
  can never break the operation it records. (Known consequence: the
  ``prune_system(metadata=True)`` record is self-erasing — it drops the
  relational DB including ``pipeline_runs``, so its own write fails and
  is swallowed by design.)
"""

from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator, Optional
from uuid import UUID, uuid4

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
        user_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
        usage: Optional[OperationUsage] = None,
        session_id: Optional[str] = None,
        background: Optional[bool] = None,
        parent_operation_id: Optional[UUID] = None,
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

    def set_user(self, user) -> None:
        """Bind the triggering user (tolerates None and partial objects)."""
        if user is None:
            return
        self.user_id = getattr(user, "id", None)
        self.tenant_id = getattr(user, "tenant_id", None)

    def set_dataset(self, dataset_id: Optional[UUID]) -> None:
        """Bind the target dataset, when the operation has exactly one."""
        self.dataset_id = dataset_id

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Bind the active session-cache id (joins SessionModelUsage)."""
        if session_id:
            self.session_id = session_id

    def set_background(self, background: bool) -> None:
        """Mark whether this call launched background work.

        True means outcome="succeeded" records "accepted and started", not
        "background work finished" (SDK-399 background-launch semantics).
        """
        self.background = background


async def _write_operation_row(
    context: OperationContext,
    started_at: datetime,
    outcome: str,
    error_class: Optional[str],
    error_message: Optional[str],
) -> None:
    from cognee.modules.pipelines.models import PipelineRun

    pipeline_run = PipelineRun(
        status=None,
        pipeline_run_id=context.operation_id,
        pipeline_name=None,
        pipeline_id=None,
        dataset_id=context.dataset_id,
        run_info=None,
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
    dataset_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    background: Optional[bool] = None,
) -> AsyncIterator[OperationContext]:
    """Record one non-pipeline operation as a single ``pipeline_runs`` row."""
    from cognee.modules.pipelines.models import OperationOutcome

    started_at = datetime.now(timezone.utc)

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
        error_class: Optional[str] = None
        error_message: Optional[str] = None

        # Eval capture (SDK-529). Imported lazily so ``import cognee`` never pays
        # for the capture package; is_active() is one global read once
        # initialized, and the OFF path constructs no scope at all. No drain()
        # here: this wraps the user-facing recall/search path, and the manifest
        # lands within FLUSH_INTERVAL_S like any other event.
        from cognee.modules.observability import capture as eval_capture

        capture_scope_cm = (
            eval_capture.run_scope(context.operation_id, dataset_id, kind="operation")
            if eval_capture.is_active()
            else nullcontext()
        )
        with capture_scope_cm as capture_scope:
            try:
                with parent_run_scope(context.operation_id):
                    yield context
            except BaseException as exc:
                outcome = OperationOutcome.FAILED
                error_class = type(exc).__name__
                error_message = scrub_error_message(exc)
                raise
            finally:
                if capture_scope is not None:
                    # Callers bind the dataset lazily via context.set_dataset().
                    capture_scope.set_dataset(context.dataset_id)
                _current_operation.reset(context_token)
                try:
                    await _write_operation_row(
                        context, started_at, outcome.value, error_class, error_message
                    )
                except Exception as write_error:
                    logger.warning(
                        "record_operation: failed to persist %s record (%s)",
                        operation_name,
                        write_error,
                    )
