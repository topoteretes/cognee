"""The values that cross the stages of one review run.

``ReviewScope`` is what the reader produces and the reviewer and writer both
read; a ``ReviewBatch`` is one unit of work the reviewer yields and the writer
persists. Every stage mutates the shared ``ReviewScope`` rather than copying it,
so a later call sees the marks and conflicts an earlier write left behind.
"""

from dataclasses import dataclass, field
from datetime import datetime

from cognee.modules.engine.models.FactConflict import FactConflict


@dataclass
class ReviewScope:
    """Everything one run may look at: its subjects, their facts, the conflicts stored
    about them, and every other node the read turned up.

    Shared, not copied. The writer updates ``facts`` and ``conflicts`` in place after
    each batch lands, so the next review call sees what the last write left behind.
    """

    dataset_id: str
    entities: dict[str, dict]
    facts: dict[str, dict]
    conflicts: dict[str, dict] = field(default_factory=dict)
    drop_conflict_ids: list[str] = field(default_factory=list)
    since: datetime | None = None
    nodes: dict[str, dict] = field(default_factory=dict)


@dataclass
class AcceptedConflict:
    """One conflict the review reported, with the status it gave each of its facts."""

    conflict: FactConflict
    fact_statuses: dict[str, str]


@dataclass
class ReviewBatch:
    """One call's accepted output, or the run's final marker."""

    scope: ReviewScope
    descriptions: dict[str, str] = field(default_factory=dict)
    conflicts: list[AcceptedConflict] = field(default_factory=list)
    dropped_conflict_ids: list[str] = field(default_factory=list)
    shown_fact_ids: set[str] = field(default_factory=set)
    final: bool = False
    # On a normal batch the subjects this call did not review; on the final
    # batch the union for the whole run, which is the only one the writer reads.
    unreviewed_entity_ids: list[str] = field(default_factory=list)
