"""Provider-neutral, durable sync progress; reporting never changes ingestion outcomes.

One PipelineRun records each invocation. Progress describes a phase, not a made-up
whole-job percentage. Nonterminal records without a local reporter are last observed
state only: another process may still own them, or their worker may have exited.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt
from pydantic.alias_generators import to_camel
from sqlalchemy import func, select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.pipelines.models.PipelineRun import PipelineRun

# The structlog exception wrapper can attach the current exception even with
# exc_info=False. Reporting errors must never log sink payloads or database values.
logger = logging.getLogger(__name__)
OPERATION_NAME = "integration_sync_progress"
SyncPhase = Literal[
    "queued", "checking", "downloading", "discovering", "processing", "indexing", "finalizing"
]
SyncState = Literal["running", "waiting", "succeeded", "partial", "failed", "cancelled", "blocked"]
SyncReason = Literal[
    "rate_limited",
    "waiting_for_worker",
    "waiting_for_lock",
    "resource_limit",
    "provider_error",
    "processing_error",
    "cancelled",
    "connection_changed",
    "incomplete",
]
SyncUnit = Literal["items", "files", "bytes", "documents", "channels", "repositories", "scopes"]
TERMINAL_STATES = frozenset({"succeeded", "partial", "failed", "cancelled", "blocked"})
# Minimum gap between database writes for observations within one phase and state.
PUBLISH_INTERVAL_SECONDS = 1.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SyncProgress(BaseModel):
    """Last observation plus phase-local estimates; no source text or exception messages."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    run_id: UUID
    provider: str
    credential_id: UUID
    dataset_id: UUID | None = None
    state: SyncState = "running"
    phase: SyncPhase = "checking"
    started_at: datetime
    phase_started_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None
    completed: StrictInt | None = Field(default=None, ge=0)
    total: StrictInt | None = Field(default=None, ge=0)
    unit: SyncUnit | None = None
    counts: dict[str, StrictInt] = Field(default_factory=dict)
    reason: SyncReason | None = None
    elapsed_seconds: float = 0
    phase_elapsed_seconds: float = 0
    phase_percent: float | None = None
    phase_eta_seconds: float | None = None
    execution_active: bool | None = None


class SyncProgressResult(BaseModel):
    """Latest invocation and all locally confirmed active invocations.

    Unknown activity counts unresolved durable runs owned by a different process or
    an exited worker. It is deliberately not reported as a live or failed job count.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    latest: SyncProgress | None = None
    active_runs: list[SyncProgress] = Field(default_factory=list)
    unknown_activity_count: int = 0


ProgressSink = Callable[[SyncProgress], Awaitable[None]]
_current: ContextVar["SyncProgressReporter | None"] = ContextVar("sync_progress", default=None)
# Runs of this process only: track_sync adds and removes its reporter, all on one loop.
_active: dict[UUID, "SyncProgressReporter"] = {}


def get_current_sync_progress() -> "SyncProgressReporter | None":
    return _current.get()


class SyncProgressReporter:
    """Keeps live progress and publishes the latest snapshot through one background writer."""

    def __init__(
        self,
        provider: str,
        credential: IntegrationCredential,
        dataset_id: UUID | None = None,
        sink: ProgressSink | None = None,
    ):
        if credential.provider != provider:
            raise ValueError("Progress provider must match the connection")
        now = _now()
        self.progress = SyncProgress(
            run_id=uuid4(),
            provider=provider,
            credential_id=credential.id,
            dataset_id=dataset_id,
            started_at=now,
            phase_started_at=now,
            updated_at=now,
        )
        self.user_id = credential.user_id
        self.workspace_id = credential.workspace_id
        self.account_id = credential.provider_account_id
        self.sink = sink
        self._lock = asyncio.Lock()
        self._sample: tuple[int, datetime] | None = None
        self._pending: SyncProgress | None = None
        self._writer: asyncio.Task[None] | None = None
        self._published: tuple[SyncPhase, SyncState] | None = None
        self._published_at = 0.0
        self._urgent = asyncio.Event()

    def snapshot(self) -> SyncProgress:
        result = self.progress.model_copy(deep=True)
        terminal = result.state in TERMINAL_STATES
        active = _active.get(result.run_id) is self and not terminal
        result.execution_active = False if terminal else True if active else None
        clock = _now() if active else result.ended_at or result.updated_at
        result.elapsed_seconds = max(0, (clock - result.started_at).total_seconds())
        result.phase_elapsed_seconds = max(0, (clock - result.phase_started_at).total_seconds())
        return result

    async def report(
        self,
        *,
        phase: SyncPhase | None = None,
        completed: int | None = None,
        total: int | None = None,
        unit: SyncUnit | None = None,
        counts: dict[str, int] | None = None,
        state: SyncState | None = None,
        reason: SyncReason | None = None,
    ) -> None:
        """Validate input, then publish best effort. Finished runs are immutable.

        A phase change resets counters, unit, timing and estimates; within a phase omitted
        values keep their last observation. Counts are cumulative developer-defined names.
        """
        async with self._lock:
            if self.progress.state in TERMINAL_STATES:
                return
            now = _now()
            values = self.progress.model_dump()
            if phase is not None and phase != self.progress.phase:
                values.update(
                    phase=phase,
                    phase_started_at=now,
                    completed=None,
                    total=None,
                    unit=None,
                    counts={},
                    phase_percent=None,
                    phase_eta_seconds=None,
                )
            for key, value in (
                ("completed", completed),
                ("total", total),
                ("unit", unit),
                ("counts", counts),
                ("state", state),
            ):
                if value is not None:
                    values[key] = value
            phase_changed = phase is not None and phase != self.progress.phase
            if reason is not None or state is not None or phase_changed:
                values["reason"] = reason
            candidate = SyncProgress.model_validate(values)
            if any(value < 0 for value in candidate.counts.values()):
                raise ValueError("Progress counts must not be negative")
            if (
                candidate.total is not None
                and candidate.completed is not None
                and candidate.completed > candidate.total
            ):
                raise ValueError("Completed progress cannot exceed the known total")
            if candidate == self.progress:
                return
            candidate.updated_at = now
            if phase_changed or candidate.state == "waiting" or self.progress.state == "waiting":
                self._sample = None
                candidate.phase_eta_seconds = None
            candidate.phase_percent = (
                100 * candidate.completed / candidate.total
                if candidate.total and candidate.completed is not None
                else None
            )
            if completed is not None and candidate.state == "running":
                if self._sample is not None:
                    previous, observed_at = self._sample
                    seconds = (now - observed_at).total_seconds()
                    delta = completed - previous
                    candidate.phase_eta_seconds = (
                        (candidate.total - completed) * seconds / delta
                        if candidate.total is not None and delta > 0 and seconds > 0
                        else None
                    )
                self._sample = (completed, now)
            if candidate.state in TERMINAL_STATES:
                candidate.ended_at = now
                candidate.phase_eta_seconds = None
                if candidate.state != "succeeded":
                    candidate.phase_percent = None
            self.progress = candidate
            self._pending = self.snapshot()
            if (candidate.phase, candidate.state) != self._published:
                self._urgent.set()
            if self._writer is None or self._writer.done():
                self._writer = asyncio.create_task(self._drain())

    async def finish(
        self,
        state: SyncState = "succeeded",
        *,
        counts: dict[str, int] | None = None,
        reason: SyncReason | None = None,
    ) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError("finish requires a terminal sync state")
        await self.report(state=state, counts=counts, reason=reason)
        await self.flush()

    async def flush(self) -> None:
        """Wait for delivery, retaining cancellation until the writer has been joined."""
        cancelled = False
        while self._writer is not None:
            writer = self._writer
            try:
                await asyncio.shield(writer)
            except asyncio.CancelledError:
                cancelled = True
                if not writer.done():
                    continue
            if self._writer is writer:
                break
        if cancelled:
            raise asyncio.CancelledError

    async def _drain(self) -> None:
        # This is a latest-state record, not an event log. Coalescing observations
        # keeps slow telemetry storage off the ingestion path without queuing items.
        # Within one phase and state, writes are spaced by PUBLISH_INTERVAL_SECONDS; a
        # phase or state change (terminal ones included) is published at once.
        loop = asyncio.get_running_loop()
        while self._pending is not None:
            wait = self._published_at + PUBLISH_INTERVAL_SECONDS - loop.time()
            if wait > 0 and not self._urgent.is_set():
                try:
                    await asyncio.wait_for(self._urgent.wait(), wait)
                except asyncio.TimeoutError:
                    pass
            self._urgent.clear()
            snapshot, self._pending = self._pending, None
            self._published = (snapshot.phase, snapshot.state)
            self._published_at = loop.time()
            await self._publish(snapshot)

    async def _publish(self, snapshot: SyncProgress | None = None) -> None:
        snapshot = self.snapshot() if snapshot is None else snapshot
        try:
            async with get_relational_engine().get_async_session() as db:
                row = await db.get(PipelineRun, snapshot.run_id)
                values = {
                    "run_info": {"sync_progress": snapshot.model_dump(mode="json")},
                    "ended_at": snapshot.ended_at,
                    "outcome": (
                        "succeeded"
                        if snapshot.state == "succeeded"
                        else "failed"
                        if snapshot.state in TERMINAL_STATES
                        else None
                    ),
                }
                if row is None:
                    db.add(
                        PipelineRun(
                            id=snapshot.run_id,
                            pipeline_run_id=snapshot.run_id,
                            user_id=self.user_id,
                            tenant_id=self.workspace_id,
                            operation_name=OPERATION_NAME,
                            started_at=snapshot.started_at,
                            dataset_id=snapshot.dataset_id,
                            origin="background",
                            **values,
                        )
                    )
                else:
                    await db.execute(
                        update(PipelineRun)
                        .where(PipelineRun.id == snapshot.run_id)
                        .values(**values)
                    )
                await db.commit()
        except Exception:
            logger.exception("Persisting integration sync progress failed", exc_info=False)
        if self.sink is not None:
            try:
                await self.sink(snapshot)
            except Exception:
                logger.exception("Delivering integration sync progress failed", exc_info=False)


@asynccontextmanager
async def track_sync(
    provider: str,
    credential: IntegrationCredential,
    *,
    dataset_id: UUID | None = None,
    sink: ProgressSink | None = None,
) -> AsyncIterator[SyncProgressReporter]:
    """Track an actual sync invocation, retaining its original error/cancellation.

    Call ``finish('partial', ...)`` for handled failures. No explicit finish means
    success on normal context exit. ContextVar propagation includes child async tasks;
    their updates after completion cannot reopen a finished run.
    """
    reporter = SyncProgressReporter(provider, credential, dataset_id, sink)
    token = _current.set(reporter)
    _active[reporter.progress.run_id] = reporter
    try:
        try:
            await reporter._publish()
            yield reporter
        except asyncio.CancelledError:
            await reporter.finish("cancelled", reason="cancelled")
            raise
        except BaseException:
            await reporter.finish("failed", reason="processing_error")
            raise
        else:
            await reporter.finish()
    finally:
        _active.pop(reporter.progress.run_id, None)
        _current.reset(token)


async def report_sync_progress(
    *,
    phase: SyncPhase | None = None,
    completed: int | None = None,
    total: int | None = None,
    unit: SyncUnit | None = None,
    counts: dict[str, int] | None = None,
    state: SyncState | None = None,
    reason: SyncReason | None = None,
) -> None:
    """Report within ``track_sync``; standalone connector calls remain unchanged."""
    reporter = get_current_sync_progress()
    if reporter is not None:
        await reporter.report(
            phase=phase,
            completed=completed,
            total=total,
            unit=unit,
            counts=counts,
            state=state,
            reason=reason,
        )


def _stored_snapshot(row: PipelineRun) -> SyncProgress:
    progress = SyncProgress.model_validate(row.run_info["sync_progress"])
    reporter = _active.get(progress.run_id)
    if reporter is not None:
        return reporter.snapshot()
    progress.execution_active = False if progress.state in TERMINAL_STATES else None
    progress.phase_eta_seconds = None
    return progress


async def get_sync_progress(credential: IntegrationCredential) -> SyncProgressResult:
    """Read only runs belonging to this currently active credential and both owners.

    Same-owner reconnects reuse the credential row: they retain its history. A change
    of either owner never exposes the prior owner's runs, including racing reconnects.
    """
    current = IntegrationCredential
    guard = (
        current.id == credential.id,
        current.provider == credential.provider,
        current.provider_account_id == credential.provider_account_id,
        current.user_id == credential.user_id,
        current.workspace_id == credential.workspace_id,
        current.status == "active",
    )
    async with get_relational_engine().get_async_session() as db:
        exists = await db.scalar(select(current.id).where(*guard))
        if exists is None:
            raise CredentialInactiveError("The connection is no longer active for this owner")
        query = (
            select(PipelineRun)
            .join(current, current.id == credential.id)
            .where(
                *guard,
                PipelineRun.operation_name == OPERATION_NAME,
                PipelineRun.user_id == credential.user_id,
                PipelineRun.tenant_id == credential.workspace_id,
                PipelineRun.run_info["sync_progress"]["credential_id"].as_string()
                == str(credential.id),
            )
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        )
        latest = (await db.execute(query.limit(1))).scalar_one_or_none()
        unresolved = query.where(PipelineRun.ended_at.is_(None))
        local_ids = tuple(_active)
        unknown = unresolved.where(PipelineRun.id.not_in(local_ids)) if local_ids else unresolved
        unknown_count = await db.scalar(
            unknown.with_only_columns(func.count(), maintain_column_froms=True).order_by(None)
        )
        local_rows = (
            (await db.execute(unresolved.where(PipelineRun.id.in_(local_ids)))).scalars().all()
            if local_ids
            else []
        )
        snapshots = [_stored_snapshot(row) for row in local_rows]
    return SyncProgressResult(
        latest=_stored_snapshot(latest) if latest is not None else None,
        active_runs=[item for item in snapshots if item.execution_active is True],
        unknown_activity_count=unknown_count or 0,
    )
