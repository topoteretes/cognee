"""The only description of the improve stages (plan Part 5.5).

``DEFAULT_STAGES`` lists the stages in execution order. The order
is load-bearing — stage 4's lessons are what stage 5 gates on, stage 5's
accepted lessons are stage 7's anchors, and stage 7 runs before conflict
review — and ``test_registry_order`` pins it; this list is the single place it
is declared. ``operations_catalog`` generates its improve rows from this list,
so the schema view cannot drift.
"""

from collections.abc import Iterable, Sequence

from cognee.shared.logging_utils import get_logger

from .stage import BaseStage
from .stages import (
    BuildTruthSubspaceStage,
    DistillSessionsStage,
    ExtractAgentContextStage,
    FeedbackWeightsStage,
    GlobalContextIndexStage,
    PersistAgentTracesStage,
    PersistSessionQAStage,
    ReviewConflictsStage,
    UpdateUserPreferencesStage,
)

DEFAULT_STAGES: list[BaseStage] = [
    FeedbackWeightsStage(),
    PersistSessionQAStage(),
    PersistAgentTracesStage(),
    ExtractAgentContextStage(),
    DistillSessionsStage(),
    UpdateUserPreferencesStage(),
    BuildTruthSubspaceStage(),
    ReviewConflictsStage(),
    GlobalContextIndexStage(),
]

# Stages that existed in earlier releases. ``IMPROVE_STAGES_DISABLED`` may
# still name one; it is ignored with a warning instead of failing the boot.
REMOVED_STAGE_NAMES = frozenset({"triplet_enrichment"})

logger = get_logger("improve")


def stage_names(stages: Iterable[BaseStage] = DEFAULT_STAGES) -> list[str]:
    return [stage.name for stage in stages]


def validate_stages(stages: Sequence[BaseStage] = DEFAULT_STAGES) -> None:
    """Raise ``ValueError`` on a nameless or duplicate stage."""
    seen: set[str] = set()
    for index, stage in enumerate(stages):
        if not stage.name:
            raise ValueError(f"stage at position {index} has no name")
        if stage.name in seen:
            raise ValueError(f"duplicate improve stage name: {stage.name!r}")
        seen.add(stage.name)


def validate_fatal_stage_policy(stages: Sequence[BaseStage] = DEFAULT_STAGES) -> None:
    """Exactly ``persist_session_qa`` is fatal — the product decision (D2).

    Losing session Q&A would be data loss, so that stage stops the run;
    every other stage must fail open.
    """
    fatal = [stage.name for stage in stages if stage.fatal]
    if fatal != ["persist_session_qa"]:
        raise ValueError(f"exactly one fatal stage (persist_session_qa) is allowed, got {fatal}")


def validate_stages_disabled(
    disabled: Iterable[str], stages: Sequence[BaseStage] = DEFAULT_STAGES
) -> None:
    """Raise ``ValueError`` unless every name in ``IMPROVE_STAGES_DISABLED`` is honourable.

    A typo would otherwise disable nothing, silently; and the ``fatal`` stage
    guards against data loss (decision D2), so configuration must not offer a
    runtime bypass of the invariant the registry enforces at import time.
    """
    disabled = list(disabled)
    known = {stage.name for stage in stages}
    removed = sorted(name for name in disabled if name in REMOVED_STAGE_NAMES and name not in known)
    if removed:
        logger.warning("IMPROVE_STAGES_DISABLED names removed stage(s) %s; ignoring them", removed)
    disabled = [name for name in disabled if name and name not in removed]
    if not disabled:
        return

    unknown = sorted(name for name in disabled if name not in known)
    if unknown:
        raise ValueError(
            f"IMPROVE_STAGES_DISABLED names unknown stage(s) {unknown}; "
            f"valid names: {sorted(known)}"
        )

    fatal = sorted(name for name in disabled if next(s for s in stages if s.name == name).fatal)
    if fatal:
        raise ValueError(
            f"IMPROVE_STAGES_DISABLED cannot disable fatal stage(s) {fatal}: "
            "skipping them would silently lose session data"
        )


validate_stages(DEFAULT_STAGES)
validate_fatal_stage_policy(DEFAULT_STAGES)
