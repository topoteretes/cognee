"""Decide how much of the LLM's response to a review call may be written, and as what.

Only validated labels of this call reach the writer. A label the call did not
show, a repeated label or a conflict about a subject this call never asked
about rejects the whole response: the caller discards it and retries on a
smaller call. A subject the response simply left out is not an error — it is
left unreviewed and selected again on the next run.
"""

from datetime import datetime

from cognee.modules.engine.models.FactConflict import FactConflict
from cognee.modules.engine.utils import generate_node_name
from cognee.shared.logging_utils import get_logger

from .build_review_prompt import ReviewPrompt
from .facts import (
    CHUNK_STATEMENT,
    NO_DATE,
    TIMESTAMP,
    get_endpoints,
    is_statement,
    parse_date,
    select_supporting_facts,
)
from .models import AcceptedConflict, ConflictsToWrite, ReviewScope
from .schema import ReportedConflict, ReviewResponse

logger = get_logger("review_conflicts")


def accept_review_response(
    response: ReviewResponse, scope: ReviewScope, prompt: ReviewPrompt
) -> ConflictsToWrite:
    """Resolve a response's labels against the prompt that was sent."""
    reviewed = _resolve_reviewed_subjects(response.reviewed, prompt)
    # Logged before the two steps below, either of which can reject the whole response:
    # a partial response is worth its warning even when the rest of it is invalid.
    unreviewed = _find_unreviewed_subjects(prompt, reviewed)
    shown_fact_ids = set(prompt.fact_labels.values())
    accepted_conflicts = _accept_conflicts(response.conflicts, scope, prompt, reviewed)
    drops = _accept_drops(
        response.drop, scope, prompt, reviewed, accepted_conflicts, shown_fact_ids
    )
    drops += _retire_orphaned_user_decisions(scope, prompt, reviewed, accepted_conflicts, drops)
    return ConflictsToWrite(
        scope=scope,
        accepted_conflicts=accepted_conflicts,
        dropped_conflict_ids=drops,
        shown_fact_ids=shown_fact_ids,
        unreviewed_entity_ids=unreviewed,
    )


def _resolve_reviewed_subjects(labels: list[str], prompt: ReviewPrompt) -> set[str]:
    """The subjects the response says it reviewed, each one a subject of this call."""
    reviewed: set[str] = set()
    for label in labels:
        subject = prompt.node_labels.get(label)
        if subject not in prompt.subject_ids or subject in reviewed:
            raise ValueError("Unknown or repeated reviewed subject")
        reviewed.add(subject)
    if not reviewed:
        raise ValueError("A review requires at least one reviewed subject")
    return reviewed


def _find_unreviewed_subjects(prompt: ReviewPrompt, reviewed: set[str]) -> list[str]:
    """Subjects the response left out.

    Nothing this call says about an unreviewed subject is written, and the next
    review selects it again.
    """
    unreviewed = [subject for subject in prompt.subject_ids if subject not in reviewed]
    if unreviewed:
        logger.warning(
            "Conflict review returned %d of %d subjects; not reviewing %s",
            len(reviewed),
            len(prompt.subject_ids),
            unreviewed,
        )
    return unreviewed


# --- conflicts ---------------------------------------------------------------


def _accept_conflicts(
    reported: list[ReportedConflict],
    scope: ReviewScope,
    prompt: ReviewPrompt,
    reviewed: set[str],
) -> list[AcceptedConflict]:
    accepted: list[AcceptedConflict] = []
    identities: set[str] = set()
    for result in reported:
        about = prompt.node_labels.get(result.about)
        if about not in prompt.subject_ids:
            raise ValueError("A conflict must belong to a subject of this call")
        if about not in reviewed:
            continue
        item = _accept_conflict(result, about, scope, prompt)
        if item is None:
            continue
        identity = str(item.conflict.id)
        if identity in identities:
            raise ValueError("Duplicate conflict identity")
        identities.add(identity)
        accepted.append(item)
    return accepted


def _accept_conflict(
    result: ReportedConflict, about: str, scope: ReviewScope, prompt: ReviewPrompt
) -> AcceptedConflict | None:
    """One reported conflict, or None when no two values are left to compare."""
    statuses = _resolve_fact_statuses(result, about, scope, prompt)
    # Identity is decided from the facts the LLM grouped, before the date filter
    # narrows them. The attribute is part of the conflict's node id, so deriving
    # it from the narrowed set would write a second node beside the one it means
    # to replace.
    attribute = _choose_conflict_attribute(result, [scope.facts[fact_id] for fact_id in statuses])
    statuses = _exclude_date_values(about, statuses, scope)
    if statuses is None:
        return None

    values = _collect_conflict_values(about, statuses, scope)
    facts = [scope.facts[fact_id] for fact_id in statuses]
    text = " ".join(result.text.split())
    reversed_dates = _is_ordering_reversed_by_dates(facts, statuses)
    if result.kind == "fixed" or reversed_dates:
        statuses = dict.fromkeys(statuses, "conflicting")
    if reversed_dates:
        text += " Supplied dates disagree with the proposed ordering."

    conflict = FactConflict(
        dataset_id=scope.dataset_id,
        about_id=about,
        attribute=attribute,
        kind=result.kind,
        status="unresolved" if "conflicting" in statuses.values() else "resolved",
        text=text,
        values=sorted(values),
        sources=sorted({source["chunk_id"] for fact in facts for source in fact["sources"]}),
    )
    return AcceptedConflict(conflict, statuses)


def _resolve_fact_statuses(
    result: ReportedConflict, about: str, scope: ReviewScope, prompt: ReviewPrompt
) -> dict[str, str]:
    """Each labelled fact in exactly one status.

    A label must be one this call showed, in any subject's section, and its fact must
    touch the conflict's subject; any other label rejects the response.
    """
    statuses: dict[str, str] = {}
    for status, labels in (
        ("current", result.current),
        ("superseded", result.superseded),
        ("conflicting", result.conflicting),
    ):
        for label in labels:
            fact_id = prompt.fact_labels.get(label)
            if fact_id is None or fact_id in statuses:
                raise ValueError("Unknown or repeated conflict fact")
            if about not in get_endpoints(scope.facts[fact_id]):
                raise ValueError("Conflict fact is unrelated to its subject")
            statuses[fact_id] = status
    if not statuses:
        raise ValueError("A conflict requires facts")
    return statuses


def _choose_conflict_attribute(result: ReportedConflict, facts: list[dict]) -> str:
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


def _collect_conflict_values(about: str, statuses: dict[str, str], scope: ReviewScope) -> set[str]:
    """The competing values: every endpoint of a labelled claim that is not the subject."""
    return {
        endpoint
        for fact_id in statuses
        if not is_statement(scope.facts[fact_id])
        for endpoint in get_endpoints(scope.facts[fact_id])
        if endpoint != about
    }


def _is_date_node(node: dict) -> bool:
    """A Timestamp or a bare ISO date names when a fact holds, never a thing the fact is about."""
    if node.get("type") == TIMESTAMP:
        return True
    try:
        datetime.fromisoformat(str(node.get("name")))
    except (TypeError, ValueError):
        return False
    return True


def _exclude_date_values(
    about: str, statuses: dict[str, str], scope: ReviewScope
) -> dict[str, str] | None:
    """Exclude the facts whose value is a Timestamp or a bare date, unless every value is one.

    Extraction turns "as of <date>, X is Y" into an edge to a date entity under the
    attribute's own relationship name. A date beside a value of any other kind is when
    the fact holds, not an alternative to it. None means that, once the dates are
    excluded, fewer than two values are left to call a conflict.
    """
    values = _collect_conflict_values(about, statuses, scope)
    dated = {value for value in values if _is_date_node(scope.nodes.get(value, {}))}
    if not dated or dated == values:
        return statuses
    kept = {
        fact_id: status
        for fact_id, status in statuses.items()
        if not dated.intersection(get_endpoints(scope.facts[fact_id]))
    }
    if len(values - dated) < 2 or not kept:
        return None
    return kept


def _is_ordering_reversed_by_dates(facts: list[dict], statuses: dict[str, str]) -> bool:
    """True when a superseded fact is dated after the current one."""
    latest = {
        status: max(
            (
                parse_date(fact.get("effective_date"))
                for fact in facts
                if statuses[fact["id"]] == status
            ),
            default=NO_DATE,
        )
        for status in ("current", "superseded")
    }
    return latest["current"] != NO_DATE and latest["superseded"] > latest["current"]


# --- drops -------------------------------------------------------------------


def _retire_orphaned_user_decisions(
    scope: ReviewScope,
    prompt: ReviewPrompt,
    reviewed: set[str],
    accepted: list[AcceptedConflict],
    drops: list[str],
) -> list[str]:
    """Drop the user decisions whose chosen fact no longer exists.

    When the document behind the fact a user chose is forgotten, the decision must go
    too, or retrieval keeps presenting the forgotten fact as the accepted one. The LLM
    is not reliable at this: it may drop or replace the conflict, or not mention it at
    all. This returns the decisions it left in place, for the subjects this call reviewed.
    """
    replaced = {str(item.conflict.id) for item in accepted}
    return [
        conflict_id
        for conflict_id in prompt.conflict_labels.values()
        if conflict_id not in replaced
        and conflict_id not in drops
        and str(scope.conflicts[conflict_id]["about_id"]) in reviewed
        and scope.conflicts[conflict_id].get("resolution_source") == "user"
        and scope.conflicts[conflict_id].get("selected_fact_id") not in scope.facts
    ]


def _accept_drops(
    labels: list[str],
    scope: ReviewScope,
    prompt: ReviewPrompt,
    reviewed: set[str],
    accepted: list[AcceptedConflict],
    shown_fact_ids: set[str],
) -> list[str]:
    """Stored conflicts to drop, once every fact still supporting them was shown."""
    replaced = {str(item.conflict.id) for item in accepted}
    drops: list[str] = []
    for label in labels:
        conflict_id = prompt.conflict_labels.get(label)
        if conflict_id is None or conflict_id in replaced or conflict_id in drops:
            raise ValueError("Unknown, repeated, or replaced drop")
        conflict = scope.conflicts[conflict_id]
        if str(conflict["about_id"]) not in prompt.subject_ids:
            raise ValueError("A dropped conflict must belong to a subject of this call")
        if str(conflict["about_id"]) not in reviewed:
            continue
        still_supporting = select_supporting_facts(scope.facts, scope.facts, [conflict])
        if still_supporting <= shown_fact_ids:
            drops.append(conflict_id)
    return drops
