"""The one shape every improve stage implements (plan Part 5.2).

A stage is three things: a gate, a call into code that already exists, and a
mapping of that code's result onto ``StageResult``. It is not an executor —
ordering, retries and task semantics stay in the pipeline layer. The registry
order is load-bearing; ``after`` names the stages that must precede this one
and ``registry.validate_stage_order`` checks it.
"""

from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

from .inputs import ImproveRunInputs
from .result import REASON_DISABLED_BY_CONFIG, REASON_NO_SESSION_IDS, StageResult

StageKind = Literal["session", "graph"]


@runtime_checkable
class ImproveStage(Protocol):
    name: str
    kind: StageKind
    fatal: bool
    after: Tuple[str, ...]
    label: str
    summary: str
    effects: List[Dict[str, Any]]
    pipeline_name: Optional[str]

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]: ...

    async def run(self, inputs: ImproveRunInputs) -> StageResult: ...


class BaseStage:
    """Default metadata and no-op gate; concrete stages override what they need."""

    name: str = ""
    kind: StageKind = "graph"
    fatal: bool = False
    after: Tuple[str, ...] = ()
    # Catalog metadata (``operations_catalog`` generates its improve rows from these).
    label: str = ""
    summary: str = ""
    effects: List[Dict[str, Any]] = []
    # The pipeline whose ``source_pipeline`` provenance stamps this stage's
    # output, when the stage is pipeline-backed.
    pipeline_name: Optional[str] = None

    def gate(self, inputs: ImproveRunInputs) -> Optional[str]:
        """Return a skip reason, or ``None`` to run. Must make zero LLM calls."""
        return None

    async def run(self, inputs: ImproveRunInputs) -> StageResult:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<ImproveStage {self.name}>"


def evaluate_gate(stage: ImproveStage, inputs: ImproveRunInputs) -> Optional[str]:
    """The run-level gates every stage shares, then the stage's own.

    Order: ``disabled_by_config`` (operator opt-out) -> ``no_session_ids``
    (session-kind stages with nothing to read) -> ``stage.gate(inputs)``.
    """
    if stage.name in inputs.config.stages_disabled:
        return REASON_DISABLED_BY_CONFIG
    if stage.kind == "session" and not inputs.has_sessions:
        return REASON_NO_SESSION_IDS
    return stage.gate(inputs)
