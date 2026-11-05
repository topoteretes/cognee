"""The values that cross the stages of one review run.

``ReviewScope`` is what the reader produces and the reviewer and writer both
read. The reviewer yields ``ConflictsToWrite`` for each review call whose response
passed the checks, and ``ConflictsToFinalize`` last; the writer writes each and
finalizes the run's writes at the end. The writer updates the shared ``ReviewScope``
in place rather than copying it, so a later call sees the marks and conflicts an
earlier write left.

Four words name what a run looks at. An *entity* is an Entity node the run
selected. A *subject* is an entity the reviewer gives a section, because it has
values or statements to compare or owns a stored conflict: it is what a conflict
is about (``FactConflict.about_id``) and what the prompt heads `Subject [n#]`. A
*section* is one subject's part of a call: its header and the facts it shows. A
*value* is the other end of a subject's claim.
"""

from dataclasses import dataclass, field
from datetime import datetime

from cognee.modules.engine.models.FactConflict import FactConflict


@dataclass
class ReviewScope:
    """Everything one run may look at: its entities, their facts, the stored conflicts
    that touch them, and every other node the read turned up.

    Shared, not copied. The writer updates ``facts`` and ``conflicts`` in place after
    each write lands, so the next review call sees what the last write left behind.
    """

    dataset_id: str
    entities: dict[str, dict]
    facts: dict[str, dict]
    conflicts: dict[str, dict] = field(default_factory=dict)
    conflict_ids_to_drop: list[str] = field(default_factory=list)
    since: datetime | None = None
    nodes: dict[str, dict] = field(default_factory=dict)


@dataclass
class AcceptedConflict:
    """One conflict the review reported, with the status it gave each of its facts."""

    conflict: FactConflict
    fact_statuses: dict[str, str]


@dataclass
class ConflictsToWrite:
    """What the writer writes for one review call whose response passed the checks.

    The run's first one may instead hold only the stored conflicts the reader found to drop.
    """

    scope: ReviewScope
    accepted_conflicts: list[AcceptedConflict] = field(default_factory=list)
    dropped_conflict_ids: list[str] = field(default_factory=list)
    shown_fact_ids: set[str] = field(default_factory=set)
    # The call's subjects its response left out; the reviewer reports them to its caller.
    unreviewed_entity_ids: list[str] = field(default_factory=list)


@dataclass
class ConflictsToFinalize:
    """The reviewer's last item, sent once every review call's conflicts were written.

    The writer clears the pending-write marker of each accepted conflict and deletes each
    dropped one. No conflict is in both lists: a response cannot drop a conflict it replaces,
    and only the one call that reviews a conflict's subject decides it.
    """

    accepted_conflicts: list[AcceptedConflict] = field(default_factory=list)
    dropped_conflict_ids: list[str] = field(default_factory=list)
