"""Decide how much of the model's review may be written, and as what.

Only validated, batch-local labels reach the writer. A label the call did not
show, a repeated label or a conflict about a subject this call never asked
about is a rejected answer: the caller discards the whole batch and retries on
a smaller one. A subject the answer simply left out is not an error — it is
left unreviewed and selected again on the next run.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from cognee.modules.engine.models.FactConflict import FactConflict
from cognee.modules.engine.utils import generate_node_name
from cognee.shared.logging_utils import get_logger

from .facts import (
    CHUNK_STATEMENT,
    NO_DATE,
    endpoints,
    fact_date,
    is_statement,
    supporting_fact_ids,
)
from .models import AcceptedConflict, ReviewBatch
from .schema import ReviewedConflict, ReviewOutput

if TYPE_CHECKING:  # review_facts.py imports this module, so these are annotations only.
    from .review_facts import RenderedCall, ReviewCall

logger = get_logger("review_conflicts")


def validate_review_output(
    output: ReviewOutput, call: "ReviewCall", rendered: "RenderedCall"
) -> ReviewBatch:
    """Resolve an answer's labels against the call that produced it."""
    descriptions = _reviewed_descriptions(output.descriptions, call, rendered)
    # Before the two steps below, either of which can reject the whole answer: a
    # partial review is worth recording even when the rest of it turns out invalid.
    unreviewed = _unreviewed_subjects(call, descriptions)
    shown_fact_ids = set(rendered.facts.values())
    conflicts = _accepted_conflicts(output.conflicts, call, rendered, descriptions)
    drops = _accepted_drops(output.drop, call, rendered, descriptions, conflicts, shown_fact_ids)
    return ReviewBatch(
        scope=call.scope,
        descriptions=descriptions,
        conflicts=conflicts,
        dropped_conflict_ids=drops,
        shown_fact_ids=shown_fact_ids,
        unreviewed_entity_ids=unreviewed,
    )


def _reviewed_descriptions(
    reviewed: list, call: "ReviewCall", rendered: "RenderedCall"
) -> dict[str, str]:
    """One rewritten description per subject the call asked about."""
    descriptions: dict[str, str] = {}
    for entry in reviewed:
        subject = rendered.entities.get(entry.entity)
        if subject not in call.subject_ids or subject in descriptions:
            raise ValueError("Unknown or repeated description subject")
        descriptions[subject] = entry.description
    if not descriptions:
        raise ValueError("A review requires at least one description")
    return descriptions


def _unreviewed_subjects(call: "ReviewCall", descriptions: dict[str, str]) -> list[str]:
    """Subjects the answer left out.

    An undescribed subject was not reviewed: nothing this call says about it is
    written, it is not stamped conflicts_reviewed_at, and the improve stage
    selects it again on the next run.
    """
    unreviewed = [subject for subject in call.subject_ids if subject not in descriptions]
    if unreviewed:
        logger.warning(
            "Fact review returned %d of %d descriptions; not reviewing %s",
            len(descriptions),
            len(call.subject_ids),
            unreviewed,
        )
    return unreviewed


# --- conflicts ---------------------------------------------------------------


def _accepted_conflicts(
    reported: list[ReviewedConflict],
    call: "ReviewCall",
    rendered: "RenderedCall",
    descriptions: dict[str, str],
) -> list[AcceptedConflict]:
    accepted: list[AcceptedConflict] = []
    identities: set[str] = set()
    for result in reported:
        about = rendered.entities.get(result.about)
        if about not in call.subject_ids:
            raise ValueError("A conflict must belong to a batch subject")
        if about not in descriptions:
            continue
        item = _accept_conflict(result, about, call, rendered)
        if item is None:
            continue
        identity = str(item.conflict.id)
        if identity in identities:
            raise ValueError("Duplicate conflict identity")
        identities.add(identity)
        accepted.append(item)
    return accepted


def _accept_conflict(
    result: ReviewedConflict, about: str, call: "ReviewCall", rendered: "RenderedCall"
) -> AcceptedConflict | None:
    """One reported conflict, or None when no two values are left to compare."""
    statuses = _labelled_fact_statuses(result, about, call, rendered)
    # Identity is decided from the facts the LLM grouped, before the date filter
    # narrows them. The attribute is part of the conflict's node id, so deriving
    # it from the narrowed set would write a second node beside the one it means
    # to replace.
    attribute = _conflict_attribute(result, [call.scope.facts[fact_id] for fact_id in statuses])
    values, statuses = _without_date_endpoints(about, statuses, call)
    if values is None:
        return None

    facts = [call.scope.facts[fact_id] for fact_id in statuses]
    text = " ".join(result.text.split())
    reversed_dates = _dates_reverse_the_ordering(facts, statuses)
    if result.kind == "fixed" or reversed_dates:
        statuses = dict.fromkeys(statuses, "conflicting")
    if reversed_dates:
        text += " Supplied dates disagree with the proposed ordering."

    conflict = FactConflict(
        dataset_id=call.scope.dataset_id,
        about_id=about,
        attribute=attribute,
        kind=result.kind,
        status="unresolved" if "conflicting" in statuses.values() else "resolved",
        text=text,
        values=sorted(values),
        sources=sorted({source["chunk_id"] for fact in facts for source in fact["sources"]}),
    )
    return AcceptedConflict(conflict, statuses)


def _labelled_fact_statuses(
    result: ReviewedConflict, about: str, call: "ReviewCall", rendered: "RenderedCall"
) -> dict[str, str]:
    """Each labelled fact in exactly one status, rejecting any label not shown for its subject."""
    statuses: dict[str, str] = {}
    for status, labels in (
        ("current", result.current),
        ("superseded", result.superseded),
        ("conflicting", result.conflicting),
    ):
        for label in labels:
            fact_id = rendered.facts.get(label)
            if fact_id is None or fact_id in statuses:
                raise ValueError("Unknown or repeated conflict fact")
            if about not in endpoints(call.scope.facts[fact_id]):
                raise ValueError("Conflict fact is unrelated to its subject")
            statuses[fact_id] = status
    if not statuses:
        raise ValueError("A conflict requires facts")
    return statuses


def _conflict_attribute(result: ReviewedConflict, facts: list[dict]) -> str:
    """One shared relationship name is the attribute; anything else needs the LLM's name."""
    relationships = {fact["relationship"] for fact in facts}
    attribute = (
        next(iter(relationships))
        if len(relationships) == 1 and CHUNK_STATEMENT not in relationships
        else generate_node_name(result.attribute.strip())
    )
    if not attribute:
        raise ValueError("A conflict requires an attribute")
    return attribute


def _conflict_values(about: str, statuses: dict[str, str], call: "ReviewCall") -> set[str]:
    """The competing values: every endpoint of a labelled claim that is not the subject."""
    return {
        endpoint
        for fact_id in statuses
        if not is_statement(call.scope.facts[fact_id])
        for endpoint in endpoints(call.scope.facts[fact_id])
        if endpoint != about
    }


def _is_date_name(name) -> bool:
    """A bare ISO date names when a fact holds, never a thing the fact is about."""
    try:
        datetime.fromisoformat(str(name))
    except (TypeError, ValueError):
        return False
    return True


def _without_date_endpoints(
    about: str, statuses: dict[str, str], call: "ReviewCall"
) -> tuple[set[str] | None, dict[str, str]]:
    """Drop the facts whose value is a bare date, unless every value is one.

    Extraction turns "as of <date>, X is Y" into an edge to a date entity under the
    attribute's own relationship name. A date beside a value of any other kind is when
    the fact holds, not an alternative to it. A None result means too little is left
    to call a conflict.
    """
    values = _conflict_values(about, statuses, call)
    dated = {
        value for value in values if _is_date_name(call.scope.nodes.get(value, {}).get("name"))
    }
    if not dated or dated == values:
        return values, statuses
    values -= dated
    statuses = {
        fact_id: status
        for fact_id, status in statuses.items()
        if not dated.intersection(endpoints(call.scope.facts[fact_id]))
    }
    if len(values) < 2 or not statuses:
        return None, statuses
    return values, statuses


def _dates_reverse_the_ordering(facts: list[dict], statuses: dict[str, str]) -> bool:
    """True when a superseded fact is dated after the current one."""
    latest = {
        status: max(
            (
                fact_date(fact.get("effective_date"))
                for fact in facts
                if statuses[fact["id"]] == status
            ),
            default=NO_DATE,
        )
        for status in ("current", "superseded")
    }
    return latest["current"] != NO_DATE and latest["superseded"] > latest["current"]


# --- drops -------------------------------------------------------------------


def _accepted_drops(
    labels: list[str],
    call: "ReviewCall",
    rendered: "RenderedCall",
    descriptions: dict[str, str],
    accepted: list[AcceptedConflict],
    shown_fact_ids: set[str],
) -> list[str]:
    """Stored conflicts to retire, once every fact still supporting them was shown."""
    replaced = {str(item.conflict.id) for item in accepted}
    drops: list[str] = []
    for label in labels:
        conflict_id = rendered.conflicts.get(label)
        if conflict_id is None or conflict_id in replaced or conflict_id in drops:
            raise ValueError("Unknown, repeated, or replaced drop")
        conflict = call.scope.conflicts[conflict_id]
        if str(conflict["about_id"]) not in call.subject_ids:
            raise ValueError("A dropped conflict must belong to a batch subject")
        if str(conflict["about_id"]) not in descriptions:
            continue
        if supporting_fact_ids(call.scope, conflict) <= shown_fact_ids:
            drops.append(conflict_id)
    return drops
