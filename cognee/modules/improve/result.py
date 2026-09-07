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
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunInfo,
)

StageStatus = Literal["completed", "already_completed", "skipped", "errored"]

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
    reason: Optional[str] = None  # required when skipped; informative otherwise
    error: Optional[str] = None
    counts: Dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    duration_ms: int = 0
    run: Optional[PipelineRunInfo] = None  # set when the stage is a pipeline

    # The untouched return of the wrapped pipeline call (``{dataset_id:
    # PipelineRunInfo}`` in blocking mode). Kept off the schema; the
    # orchestrator lifts stage 8's copy onto ``ImproveResult.memify_run``.
    _raw_run: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _skipped_needs_reason(self) -> "StageResult":
        if self.status == "skipped" and not self.reason:
            raise ValueError(f"stage '{self.stage}' is skipped without a reason")
        return self

    @property
    def raw_run(self) -> Any:
        return self._raw_run

    @property
    def ok(self) -> bool:
        return self.status in ("completed", "already_completed")

    @classmethod
    def skipped(cls, stage: str, reason: str) -> "StageResult":
        return cls(stage=stage, status="skipped", reason=reason)

    @classmethod
    def errored(cls, stage: str, error: Any, **counts: int) -> "StageResult":
        return cls(stage=stage, status="errored", error=_error_text(error), counts=dict(counts))

    @classmethod
    def completed(cls, stage: str, llm_calls: int = 0, **counts: int) -> "StageResult":
        return cls(stage=stage, status="completed", counts=dict(counts), llm_calls=llm_calls)

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
        error: Optional[str] = None
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


def first_run_info(run_result: Any) -> Optional[PipelineRunInfo]:
    """First ``PipelineRunInfo`` inside an executor return, if any."""
    if isinstance(run_result, PipelineRunInfo):
        return run_result
    if isinstance(run_result, dict):
        for value in run_result.values():
            if isinstance(value, PipelineRunInfo):
                return value
    if isinstance(run_result, (list, tuple)):
        for value in run_result:
            if isinstance(value, PipelineRunInfo):
                return value
    return None


def _error_text(error: Any) -> Optional[str]:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


ImproveStatus = Literal["completed", "errored", "skipped", "running"]


class ImproveResult(BaseModel):
    """One entry per stage, in registry order, for one ``improve()`` run.

    ``status`` summarises the stages: ``running`` while a background chain is
    still going, ``errored`` when any stage errored, ``skipped`` when every
    stage was skipped (a lost lock claim, an unchanged graph with nothing
    opted in), ``completed`` otherwise. ``await result.wait()`` blocks on a
    background chain and returns the same, now finished, object.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: Optional[UUID] = None
    dataset_name: Optional[str] = None
    session_ids: List[str] = Field(default_factory=list)
    stages: List[StageResult] = Field(default_factory=list)
    # Legacy: the raw return of the memify enrichment stage, nested for one
    # minor release (D4). ``{}`` when that stage did not run.
    memify_run: Any = None
    background: bool = False
    finished: bool = True
    # Set when a fatal stage aborted the chain (always raised in the
    # foreground; recorded here in background mode where a raise has nowhere
    # to go).
    error: Optional[str] = None

    _task: Optional["asyncio.Task"] = PrivateAttr(default=None)

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

    @property
    def ok(self) -> List[StageResult]:
        """Stages that did or had already done their work."""
        return [stage for stage in self.stages if stage.ok]

    @property
    def lock_held(self) -> bool:
        return bool(self.stages) and all(
            stage.status == "skipped" and stage.reason == REASON_LOCK_HELD for stage in self.stages
        )

    def stage(self, name: str) -> Optional[StageResult]:
        for stage in self.stages:
            if stage.stage == name:
                return stage
        return None

    def stage_summary(self) -> str:
        """``name=status`` pairs, comma-joined — the tracing attribute value."""
        return ",".join(f"{stage.stage}={stage.status}" for stage in self.stages)

    def to_legacy_dict(self) -> Any:
        """The pre-1.x return shape: the memify enrichment run info (or ``{}``)."""
        return self.memify_run if self.memify_run is not None else {}

    async def wait(self) -> "ImproveResult":
        """Await the background chain (no-op for foreground runs)."""
        if self._task is not None and not self._task.done():
            await asyncio.shield(self._task)
        return self

    @classmethod
    def all_skipped(
        cls,
        stage_names: List[str],
        reason: str,
        *,
        dataset_id: Optional[UUID] = None,
        dataset_name: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ) -> "ImproveResult":
        return cls(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            session_ids=list(session_ids or []),
            stages=[StageResult.skipped(name, reason) for name in stage_names],
            memify_run={},
        )
