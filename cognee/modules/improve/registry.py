"""The only description of the improve chain (plan Part 5.5).

``DEFAULT_STAGES`` lists the nine stages in the order of plan Part 2. The order
is load-bearing: stage 4's lessons are what stage 5 gates on, stage 5's
accepted lessons are stage 7's anchors, and stage 7 runs before enrichment.
Each stage may declare ``after=(...)``; ``validate_stage_order`` asserts the
list satisfies every declaration and runs at import time.

``operations_catalog`` generates its improve rows from this list, so the schema
view cannot drift. ``memify_task_registry`` is untouched: it names tasks for the
public ``memify(tasks=[...])`` API at a finer grain than a stage.
"""

from typing import Dict, Iterable, List, Sequence

from .stage import ImproveStage
from .stages import (
    BuildTruthSubspaceStage,
    DistillSessionsStage,
    ExtractAgentContextStage,
    FeedbackWeightsStage,
    GlobalContextIndexStage,
    PersistAgentTracesStage,
    PersistSessionQAStage,
    TripletEnrichmentStage,
    UpdateUserPreferencesStage,
)

DEFAULT_STAGES: List[ImproveStage] = [
    FeedbackWeightsStage(),
    PersistSessionQAStage(),
    PersistAgentTracesStage(),
    ExtractAgentContextStage(),
    DistillSessionsStage(),
    UpdateUserPreferencesStage(),
    BuildTruthSubspaceStage(),
    TripletEnrichmentStage(),
    GlobalContextIndexStage(),
]


def stage_names(stages: Iterable[ImproveStage] = DEFAULT_STAGES) -> List[str]:
    return [stage.name for stage in stages]


def validate_stage_order(stages: Sequence[ImproveStage] = DEFAULT_STAGES) -> None:
    """Raise ``ValueError`` unless every ``after`` declaration is satisfied.

    Also rejects duplicate names and an ``after`` that names an unknown stage,
    and requires exactly one ``fatal`` stage (decision D2).
    """
    positions: Dict[str, int] = {}
    for index, stage in enumerate(stages):
        if not stage.name:
            raise ValueError(f"stage at position {index} has no name")
        if stage.name in positions:
            raise ValueError(f"duplicate improve stage name: {stage.name!r}")
        positions[stage.name] = index

    for stage in stages:
        for predecessor in stage.after:
            if predecessor not in positions:
                raise ValueError(
                    f"stage {stage.name!r} declares after={predecessor!r}, which is not registered"
                )
            if positions[predecessor] >= positions[stage.name]:
                raise ValueError(
                    f"stage {stage.name!r} must run after {predecessor!r}, "
                    f"but the registry lists it first"
                )

    fatal = [stage.name for stage in stages if stage.fatal]
    if fatal != ["persist_session_qa"]:
        raise ValueError(f"exactly one fatal stage (persist_session_qa) is allowed, got {fatal}")


validate_stage_order(DEFAULT_STAGES)
