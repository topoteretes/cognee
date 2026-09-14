"""Result types for one ``improve()`` run (plan Part 5.4).

``StageResult`` is one entry per stage, on ``PipelineRunInfo``'s status
vocabulary plus ``skipped``: ``PipelineRunCompleted`` -> ``completed``,
``PipelineRunAlreadyCompleted`` -> ``already_completed``, ``PipelineRunErrored``
-> ``errored``. A pipeline-backed stage takes its status *from* its run info
(``StageResult.from_pipeline_run``) and never sets the two independently.

``ImproveResult`` holds one ``StageResult`` per stage, in registry order, and
is what every surface hands back. The legacy memify return stays reachable as
``.memify_run`` for one minor release (decision D4).
"""

import asyncio
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunInfo,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("improve")

StageStatus = Literal["completed", "already_completed", "skipped", "errored"]

# The one stage whose raw pipeline return is nested on
# ``ImproveResult.memify_run`` for one minor release (decision D4).
LEGACY_MEMIFY_STAGE_NAME = "triplet_enrichment"

# Gate reasons the orchestrator itself produces (stages add their own).
REASON_LOCK_HELD = "lock_held"
REASON_NO_SESSION_IDS = "no_session_ids"
REASON_DISABLED_BY_CONFIG = "disabled_by_config"
REASON_ABORTED_BY_FATAL_STAGE = "aborted_by_fatal_stage"
REASON_BACKEND_UNSUPPORTED = "backend_unsupported"


class StageResult(BaseModel):
    """What one stage did in one run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage: str
    status: StageStatus
    reason: str | None = None  # required when skipped; informative otherwise
    error: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    duration_ms: int = 0
    run: PipelineRunInfo | None = None  # set when the stage is a pipeline

    # The untouched return of the wrapped pipeline call (``{dataset_id:
    # PipelineRunInfo}`` in blocking mode). Kept off the schema; the
    # orchestrator lifts stage 8's copy onto ``ImproveResult.memify_run``.
    _raw_run: Any = PrivateAttr(default=None)

    # The exception a stage raised, when it did. A stage whose wrapped pipeline
    # reported PipelineRunErrored instead of raising carries none, so the
    # orchestrator can tell the two apart and re-raise the original.
    _exception: BaseException | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _skipped_needs_reason(self) -> "StageResult":
        if self.status == "skipped" and not self.reason:
            raise ValueError(f"stage '{self.stage}' is skipped without a reason")
        return self

    @property
    def raw_run(self) -> Any:
        return self._raw_run

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    @classmethod
    def skipped(cls, stage: str, reason: str) -> "StageResult":
        return cls(stage=stage, status="skipped", reason=reason)

    @classmethod
    def errored(cls, stage: str, error: Any, **counts: int) -> "StageResult":
        result = cls(stage=stage, status="errored", error=_error_text(error), counts=dict(counts))
        if isinstance(error, BaseException):
            result._exception = error
        return result

    @classmethod
    def completed(cls, stage: str, **counts: int) -> "StageResult":
        return cls(stage=stage, status="completed", counts=dict(counts))

    @classmethod
    def from_pipeline_run(cls, stage: str, run_result: Any, **counts: int) -> "StageResult":
        """Map a pipeline executor's return onto a stage status.

        ``run_result`` is what ``memify()`` / ``run_pipeline_blocking`` hand
        back: a ``{dataset_id: PipelineRunInfo}`` mapping, a bare
        ``PipelineRunInfo``, or (from callers that never reached the executor)
        something else entirely, which counts as completed with no run info.
        """
        run_info = first_run_info(run_result)
        status: StageStatus = "completed"
        error: str | None = None
        if isinstance(run_info, PipelineRunErrored):
            status = "errored"
            error = run_info.error_message or _error_text(run_info.payload)
        elif isinstance(run_info, PipelineRunAlreadyCompleted):
            status = "already_completed"
        elif isinstance(run_info, PipelineRunCompleted):
            status = "completed"
        elif isinstance(run_info, PipelineRunInfo):
            # Started / Yield / Progress: the executor returned before the end
            # (background mode). Report what we know, never invent a status.
            status = "completed"
        result = cls(stage=stage, status=status, error=error, counts=dict(counts), run=run_info)
        result._raw_run = run_result
        return result


def first_run_info(run_result: Any) -> PipelineRunInfo | None:
    """First ``PipelineRunInfo`` inside an executor return, if any.

    Executors hand back a ``{dataset_id: PipelineRunInfo}`` mapping or a bare
    ``PipelineRunInfo``; anything else carries no run info.
    """
    if isinstance(run_result, PipelineRunInfo):
        return run_result
    if isinstance(run_result, dict):
        for value in run_result.values():
            if isinstance(value, PipelineRunInfo):
                return value
    return None


def _error_text(error: Any) -> str | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def stage_detail_text(stage: Any) -> str:
    """The human-readable detail every surface shows for one stage.

    Reason when skipped, counts as ``key=value``, and the error when errored,
    "; "-joined. Accepts a ``StageResult`` or its serialized dict, so the CLI's
    remote path and MCP format the same content the in-process path does —
    the layout (columns, dashes) stays with each surface.
    """

    def field(key: str, default: Any = None) -> Any:
        if isinstance(stage, dict):
            return stage.get(key, default)
        return getattr(stage, key, default)

    details = []
    reason = field("reason")
    if reason:
        details.append(str(reason))
    counts = field("counts") or {}
    if isinstance(counts, dict) and counts:
        details.append(", ".join(f"{key}={value}" for key, value in counts.items()))
    error = field("error")
    if error and field("status") == "errored":
        details.append(str(error))
    return "; ".join(details)


ImproveStatus = Literal["completed", "errored", "skipped", "running"]


class ImproveResult(BaseModel):
    """One entry per stage, in registry order, for one ``improve()`` run.

    ``status`` summarises the stages: ``running`` while a background run is
    still going, ``errored`` when any stage errored, ``skipped`` when every
    stage was skipped (a lost lock claim, an unchanged graph with nothing
    opted in), ``completed`` otherwise. ``await result.wait()`` blocks on a
    background run and returns the same, now finished, object.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: UUID | None = None
    dataset_name: str | None = None
    session_ids: list[str] = Field(default_factory=list)
    stages: list[StageResult] = Field(default_factory=list)
    # Legacy: the raw return of the memify enrichment stage, nested for one
    # minor release (D4). ``{}`` when that stage did not run.
    memify_run: Any = None
    background: bool = False
    finished: bool = True
    # Set when a fatal stage aborted the run (always raised in the
    # foreground; recorded here in background mode where a raise has nowhere
    # to go).
    error: str | None = None

    _task: asyncio.Task | None = PrivateAttr(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> ImproveStatus:
        if not self.finished:
            return "running"
        if self.error or any(stage.status == "errored" for stage in self.stages):
            return "errored"
        if self.stages and all(stage.status == "skipped" for stage in self.stages):
            return "skipped"
        return "completed"

    def record(self, stage_result: StageResult) -> None:
        """Append one stage's outcome, nesting the legacy memify return (D4)."""
        self.stages.append(stage_result)
        if stage_result.stage == LEGACY_MEMIFY_STAGE_NAME:
            self.memify_run = stage_result.raw_run if stage_result.raw_run is not None else {}

    def stage(self, name: str) -> StageResult | None:
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    def stage_summary(self) -> str:
        """``name=status`` pairs, comma-joined — the tracing attribute value."""
        return ",".join(f"{stage.stage}={stage.status}" for stage in self.stages)

    def attach_background_task(self, task: asyncio.Task) -> "ImproveResult":
        """Bind the detached task that is filling this result, for ``wait()``.

        The task pointer is a ``PrivateAttr`` to stay off the schema; this is
        the one way to set it from outside the class.
        """
        self._task = task
        return self

    async def wait(self) -> "ImproveResult":
        """Await the background run (no-op for foreground runs)."""
        if self._task is not None and not self._task.done():
            await asyncio.shield(self._task)
        return self

    @classmethod
    def from_remote_payload(cls, payload: Any, session_ids: list[str]) -> "ImproveResult":
        """Rebuild a result handed back by a remote cognee server.

        A server running the same stages returns the serialized result; an older one
        returns the legacy memify run mapping, which is nested as
        ``memify_run`` with no stage detail.
        """
        if isinstance(payload, dict) and "stages" in payload:
            try:
                return cls.model_validate(payload)
            except Exception as error:
                logger.debug(
                    "improve: remote result did not validate as ImproveResult: %s",
                    error,
                    exc_info=True,
                )

        return cls(session_ids=list(session_ids), stages=[], memify_run=payload)

    @classmethod
    def all_skipped(
        cls,
        stage_names: list[str],
        reason: str,
        *,
        dataset_id: UUID | None = None,
        dataset_name: str | None = None,
        session_ids: list[str] | None = None,
    ) -> "ImproveResult":
        return cls(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            session_ids=list(session_ids or []),
            stages=[StageResult.skipped(name, reason) for name in stage_names],
            memify_run={},
        )
