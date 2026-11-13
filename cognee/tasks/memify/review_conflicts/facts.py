"""Readings of the fact dictionaries a review works on.

``ReviewScope.facts`` holds plain dicts straight from the graph read, shared by
fact selection, the prompt, the validator and the writer. These are the readings
they all agree on: what a fact's endpoints are, when it holds, whether it is a
chunk statement rather than an entity claim, and which facts support a stored
conflict.
"""

from collections import defaultdict
from collections.abc import Collection, Iterable
from datetime import datetime, timezone

# An edge from a document chunk to an entity: what one document states, as
# opposed to a claim linking two entities. A statement is shown only when it
# pairs with another or supports a stored conflict; it ranks last, names no
# competing value, and FactConflict.values skips its endpoints.
CHUNK_STATEMENT = "contains"

# The node type a `{verb}_at` claim points at; its name is the date.
TIMESTAMP = "Timestamp"

# The floor every undated fact sorts to, and the value that means "no date".
NO_DATE = datetime.min.replace(tzinfo=timezone.utc)


def get_endpoints(fact: dict) -> tuple[str, str]:
    return fact["source"], fact["target"]


def is_statement(fact: dict) -> bool:
    return fact["relationship"] == CHUNK_STATEMENT


def parse_date(value) -> datetime:
    """A stored date as an aware datetime; anything missing is NO_DATE."""
    if not value:
        return NO_DATE
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def is_observed_since(fact: dict, since: datetime | None) -> bool:
    """Was the fact observed, its write completed, after the last review began?"""
    return since is not None and parse_date(fact.get("observed_at")) > parse_date(since)


def has_conflict_mark(fact: dict, conflict_ids: Collection[str]) -> bool:
    """A mark a previous write left on this fact for one of those conflicts."""
    return any(
        str(mark.get("conflict_id")) in conflict_ids
        for mark in fact["properties"].get("conflict_marks", [])
    )


def _is_from_cited_chunk(fact: dict, cited_chunks_by_subject: dict[str, set[str]]) -> bool:
    """Does the fact come from a chunk that a conflict about one of its endpoints cites?"""
    cited_chunks = set().union(
        *(cited_chunks_by_subject.get(endpoint, ()) for endpoint in get_endpoints(fact))
    )
    return any(source["chunk_id"] in cited_chunks for source in fact["sources"])


def select_supporting_facts(
    facts: dict[str, dict], fact_ids: Collection[str], conflicts: Iterable[dict]
) -> set[str]:
    """Which of these facts support any of these stored conflicts.

    A fact supports a conflict when it carries the conflict's mark, or when it touches
    the conflict's subject and comes from a chunk the conflict cites. Citations count
    as well as marks, so an unfinished earlier write still holds its facts.

    The caller says which conflicts count: every stored conflict to pick and rank a
    section's facts, the subject's own to decide what a cut section keeps, and one to
    decide whether dropping it is safe. A write between calls may add marks or change
    citations, so this is read per call rather than kept with the comparisons, which a
    run never changes.
    """
    conflict_ids: set[str] = set()
    cited_chunks_by_subject: dict[str, set[str]] = defaultdict(set)
    for conflict in conflicts:
        conflict_ids.add(str(conflict["id"]))
        cited_chunks_by_subject[str(conflict["about_id"])].update(conflict.get("sources", []))
    return {
        fact_id
        for fact_id in fact_ids
        if has_conflict_mark(facts[fact_id], conflict_ids)
        or _is_from_cited_chunk(facts[fact_id], cited_chunks_by_subject)
    }
