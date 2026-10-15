"""The one shape every improve stage implements (plan Part 5.2).

A stage is three things: a gate, a call into code that already exists, and a
mapping of that code's result onto ``StageResult``. It is not an executor —
ordering, retries and task semantics stay in the pipeline layer. The registry
order is load-bearing and pinned by a test; how each stage is described to the
schema view lives with that view (``operations_catalog``), not here.

``execute_stage`` is how ``improve()`` runs one stage: the shared run-level
gates (``evaluate_gate``), then the stage's own ``run``, timed and never
raising.
"""

import time

from cognee.shared.logging_utils import get_logger

from .inputs import ImproveRunInputs
from .result import REASON_DISABLED_BY_CONFIG, REASON_NO_SESSION_IDS, StageResult

logger = get_logger("improve")


class BaseStage:
    """Default metadata and no-op gate; concrete stages override what they need."""

    name: str = ""
    # Session-fed stages are skipped with ``no_session_ids`` when the run was
    # given none; the rest work on the graph alone.
    needs_sessions: bool = False
    fatal: bool = False

    def gate(self, inputs: ImproveRunInputs) -> str | None:
        """Return a skip reason, or ``None`` to run. Must make zero LLM calls."""
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def evaluate_gate(stage: BaseStage, inputs: ImproveRunInputs) -> str | None:
    """The run-level gates every stage shares, then the stage's own.

    Order: ``disabled_by_config`` (operator opt-out) -> ``no_session_ids``
    (session-fed stages with nothing to read) -> ``stage.gate(inputs)``.
    """
    if stage.name in inputs.config.stages_disabled:
        return REASON_DISABLED_BY_CONFIG
    if stage.needs_sessions and not inputs.has_sessions:
        return REASON_NO_SESSION_IDS
    return stage.gate(inputs)


async def execute_stage(stage: BaseStage, inputs: ImproveRunInputs) -> StageResult:
    """Gate one stage, run it if the gate lets it, and time what ran.

    Never raises. A stage that raised comes back as an ``errored``
    ``StageResult`` carrying its exception, so the caller has one shape to
    branch on whether the stage raised or its wrapped pipeline merely reported
    ``PipelineRunErrored``. A gate that itself fails is treated as open — the
    stage runs — because gates exist only to save wasted work.
    """
    started_at = time.perf_counter()

    try:
        skip_reason = evaluate_gate(stage, inputs)
    except Exception as error:
        logger.warning("improve: gate for stage '%s' failed: %s", stage.name, error, exc_info=True)
        skip_reason = None

    if skip_reason is not None:
        logger.debug("improve: stage '%s' skipped (%s)", stage.name, skip_reason)
        return StageResult.skipped(stage.name, skip_reason)

    try:
        stage_result = await stage.run(inputs)
    except Exception as error:
        logger.warning("improve: stage '%s' failed: %s", stage.name, error, exc_info=True)
        stage_result = StageResult.errored(stage.name, error)

    stage_result.duration_ms = int((time.perf_counter() - started_at) * 1000)
    return stage_result
