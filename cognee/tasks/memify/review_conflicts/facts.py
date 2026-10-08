"""Readings of the fact dictionaries a review works on.

``ReviewScope.facts`` holds plain dicts straight from the graph read, shared by
the renderer, the validator and the writer. These are the readings all three
agree on: what a fact's endpoints are, when it holds, whether it is a chunk
statement rather than an entity claim, which facts compete for one slot, and
which facts a stored conflict already claims.
"""

from collections import defaultdict
from collections.abc import Collection
from datetime import datetime, timezone

from .models import ReviewScope

# An edge from a document chunk to an entity: what one document states, as
# opposed to a claim linking two entities. A statement is background — it is
# ranked last, it names no competing value, and FactConflict.values skips its
# endpoints.
CHUNK_STATEMENT = "contains"

# The floor every undated fact sorts to, and the value that means "no date".
NO_DATE = datetime.min.replace(tzinfo=timezone.utc)


def endpoints(fact: dict) -> tuple[str, str]:
    return fact["source"], fact["target"]


def is_statement(fact: dict) -> bool:
    return fact["relationship"] == CHUNK_STATEMENT


def fact_date(value) -> datetime:
    """A stored date as an aware datetime; anything missing is NO_DATE."""
    if not value:
        return NO_DATE
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def competing_groups(facts: dict[str, dict]) -> list[set[str]]:
    """Facts sharing a relationship name and one endpoint, i.e. candidates to compare.

    Keyed by (relationship, which side the shared endpoint is on, that endpoint),
    so "Acme's CEO" and "Alice's employer" stay separate groups.
    """
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for fact_id, fact in facts.items():
        if is_statement(fact):
            continue
        groups[(fact["relationship"], "source", fact["source"])].add(fact_id)
        groups[(fact["relationship"], "target", fact["target"])].add(fact_id)
    return [fact_ids for fact_ids in groups.values() if len(fact_ids) > 1]


class FactIndex:
    """The fact relationships that cannot change during a run, indexed once.

    Which facts touch a node, and which facts compete for one slot, are fixed by
    the read. Conflict support is not: the writer rewrites citations and marks
    between batches, so ``cited_fact_ids`` stays a per-call read.
    """

    def __init__(self, facts: dict[str, dict]):
        self.facts_by_endpoint: dict[str, set[str]] = defaultdict(set)
        for fact_id, fact in facts.items():
            for endpoint in endpoints(fact):
                self.facts_by_endpoint[endpoint].add(fact_id)
        self.groups = competing_groups(facts)
        self.groups_by_fact: dict[str, set[int]] = defaultdict(set)
        for index, group in enumerate(self.groups):
            for fact_id in group:
                self.groups_by_fact[fact_id].add(index)

    def facts_touching(self, node_id: str) -> set[str]:
        return self.facts_by_endpoint.get(node_id, set())

    def group_indices(self, fact_ids: Collection[str]) -> set[int]:
        """The competing groups any of these facts belongs to."""
        return set().union(*(self.groups_by_fact.get(fact_id, ()) for fact_id in fact_ids))

    def competing_with(self, fact_ids: Collection[str]) -> set[str]:
        """Every fact that shares a competing group with any of these facts."""
        return set().union(*(self.groups[index] for index in self.group_indices(fact_ids)))


def fact_priority(
    fact: dict,
    since: datetime | None,
    cited_ids: Collection[str],
    competing_ids: Collection[str],
) -> tuple[int, float, str]:
    """Sort key, best first.

    Written since the last review, then support a stored conflict already claims,
    then a fact with a competitor, then an ungrouped claim, then a chunk
    statement; within a tier the most recently dated fact wins.
    """
    if since is not None and fact_date(fact.get("observed_at")) > fact_date(since):
        tier = 0
    elif fact["id"] in cited_ids:
        tier = 1
    elif fact["id"] in competing_ids:
        tier = 2
    else:
        tier = 4 if is_statement(fact) else 3
    age = (fact_date(fact.get("effective_date")) - NO_DATE).total_seconds()
    return tier, -age, fact["id"]


def has_mark_for(fact: dict, conflict_ids: Collection[str]) -> bool:
    """A mark a previous write left on this fact for one of those conflicts."""
    return any(
        str(mark.get("conflict_id")) in conflict_ids
        for mark in fact["properties"].get("conflict_marks", [])
    )


def cited_fact_ids(scope: ReviewScope, fact_ids: Collection[str]) -> set[str]:
    """Which of these facts any stored conflict already claims.

    Bulk twin of ``supporting_fact_ids``, used for ranking. Previous yielded
    writes may have added conflict marks or changed citations, so this is read
    per call rather than cached alongside the immutable fact index.
    """
    stored_ids = set()
    cited_chunks_by_subject: dict[str, set[str]] = defaultdict(set)
    for conflict in scope.conflicts.values():
        stored_ids.add(str(conflict["id"]))
        cited_chunks_by_subject[str(conflict["about_id"])].update(conflict.get("sources", []))

    cited = set()
    for fact_id in fact_ids:
        fact = scope.facts[fact_id]
        if has_mark_for(fact, stored_ids):
            cited.add(fact_id)
            continue
        chunks = set().union(
            *(cited_chunks_by_subject.get(endpoint, ()) for endpoint in endpoints(fact))
        )
        if any(source["chunk_id"] in chunks for source in fact["sources"]):
            cited.add(fact_id)
    return cited


def supporting_fact_ids(scope: ReviewScope, conflict: dict) -> set[str]:
    """Every fact one stored conflict claims, across the whole scope.

    Per-conflict twin of ``cited_fact_ids``, used to decide whether a drop is
    safe. Includes cited support as well as marks, so an unfinished prior write
    still holds its facts.
    """
    conflict_ids = {str(conflict["id"])}
    cited_chunks = set(conflict.get("sources", []))
    about = str(conflict["about_id"])
    return {
        fact_id
        for fact_id, fact in scope.facts.items()
        if has_mark_for(fact, conflict_ids)
        or (
            about in endpoints(fact)
            and any(source["chunk_id"] in cited_chunks for source in fact["sources"])
        )
    }
