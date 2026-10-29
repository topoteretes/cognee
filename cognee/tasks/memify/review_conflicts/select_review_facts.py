"""Choose what a review shows: which entities are subjects, and which facts each section holds.

A subject is an entity whose facts offer something to compare, a value group or a
statement pair, or that owns a stored conflict. Its section holds the facts of its
groups and pairs, and its facts that support a stored conflict. Nothing here counts
tokens: for a section too large for one call, ``select_lists_to_leave_out`` says
what it may lose and in which order, and the reviewer decides how much.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from .facts import (
    TIMESTAMP,
    get_endpoints,
    has_conflict_mark,
    is_observed_since,
    is_statement,
    select_supporting_facts,
)
from .models import ReviewScope

# A value group is kept when it holds at least this many values, cited by at
# least this many distinct documents: one document rarely contradicts itself.
MIN_GROUP_VALUES = 2
MIN_GROUP_DOCUMENTS = 2

# Two statements pair when, with their literals masked, they share at least this
# many content words.
MIN_SHARED_CONTENT_WORDS = 2

# A figure (a number with its unit, a date, a code, an amount), which two paired
# statements must differ in; and a capitalised name, so that two statements naming
# the same people do not count that as shared content. The units listed only make one
# number given in two units (48 hours, 48 days) count as two figures; a number with
# any other unit is compared by its digits.
_LITERAL = re.compile(
    r"\d+(?:[.,:/-]\d+)*\s*(?:million|thousand|metres|meters|hours|days|%)?"
    r"|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"
)

# Function words are not content: two statements sharing only these do not pair.
_STOPWORDS = frozenset(
    ["the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "are", "was", "were", "by"]
    + ["with", "and", "or", "as", "its", "it", "this", "that", "from"]
)

# A relationship name linking one subject to at least this many values lists them
# (a team, a product range) rather than naming one attribute's competing values. This
# assumes an attribute's history has fewer values: a longer one, in a section too large
# for one call, is left out like a list unless a stored conflict about the subject
# marks its facts.
MIN_LIST_VALUES = 5


# --- value groups and statement pairs ----------------------------------------


def _group_values_by_shared_type(values: list[str], nodes: dict[str, dict]) -> list[set[str]]:
    """Connected components under "share a type"; the untyped values group only with each other."""
    merged: list[tuple[set[str], set[str]]] = []
    for value in values:
        own = set(nodes.get(value, {}).get("types", []))
        ids, kinds, kept = {value}, set(own), []
        for group_ids, group_types in merged:
            if own & group_types or not (own or group_types):
                ids |= group_ids
                kinds |= group_types
            else:
                kept.append((group_ids, group_types))
        merged = kept + [(ids, kinds)]
    return [ids for ids, _ in merged]


def find_value_groups(facts: dict[str, dict], nodes: dict[str, dict]) -> dict[str, list[set[str]]]:
    """Each entity's groups of claims whose values could stand for one attribute.

    Entity values are grouped when their types overlap, in either direction and
    under any relationship name; Timestamp values are grouped under one
    relationship name. A group is kept when it holds at least MIN_GROUP_VALUES
    values cited by at least MIN_GROUP_DOCUMENTS documents.
    """
    claims: dict[str, dict[str, str]] = defaultdict(dict)  # entity -> fact id -> value id
    for fact_id, fact in facts.items():
        if is_statement(fact):
            continue
        source, target = get_endpoints(fact)
        claims[source][fact_id] = target
        if nodes.get(target, {}).get("type") != TIMESTAMP:
            claims[target][fact_id] = source
    groups: dict[str, list[set[str]]] = {}
    for entity_id, value_of in claims.items():
        dated: dict[str, set[str]] = defaultdict(set)
        entity_values = set()
        for fact_id, value in value_of.items():
            if nodes.get(value, {}).get("type") == TIMESTAMP:
                dated[facts[fact_id]["relationship"]].add(fact_id)
            else:
                entity_values.add(value)
        possible_groups = [
            {fact_id for fact_id, value in value_of.items() if value in component}
            for component in _group_values_by_shared_type(sorted(entity_values), nodes)
        ] + list(dated.values())
        kept = [
            group
            for group in possible_groups
            if len({value_of[fact_id] for fact_id in group}) >= MIN_GROUP_VALUES
            and len(
                {source["data_id"] for fact_id in group for source in facts[fact_id]["sources"]}
            )
            >= MIN_GROUP_DOCUMENTS
        ]
        if kept:
            groups[entity_id] = kept
    return groups


def _extract_statement_body(fact: dict) -> str:
    """What the chunk states, after the `Document chunk mentions <name>:` prefix."""
    return (fact["properties"].get("edge_text") or "").split(":", 1)[-1]


def _extract_literals(body: str) -> set[str]:
    return {match.group(0).strip() for match in _LITERAL.finditer(body)}


def _extract_content_words(body: str) -> set[str]:
    words = (word.lower().strip(".,;:()") for word in _LITERAL.sub(" ", body).split())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def _collect_document_ids(fact: dict) -> set[str]:
    return {source["data_id"] for source in fact["sources"]}


def _extract_statement_features(fact: dict) -> tuple[set[str], set[str], set[str]]:
    """The documents stating it, the figures it gives and the content words around them."""
    body = _extract_statement_body(fact)
    figures = {literal for literal in _extract_literals(body) if literal[0].isdigit()}
    return _collect_document_ids(fact), figures, _extract_content_words(body)


def _is_statement_pair(first: tuple, second: tuple) -> bool:
    """Different documents, a figure in each that differs, enough shared content words."""
    first_documents, first_figures, first_words = first
    second_documents, second_figures, second_words = second
    return (
        bool(first_documents and second_documents)
        and first_documents.isdisjoint(second_documents)
        and bool(first_figures and second_figures)
        and first_figures != second_figures
        and len(first_words & second_words) >= MIN_SHARED_CONTENT_WORDS
    )


def find_statement_pairs(facts: dict[str, dict]) -> dict[str, list[set[str]]]:
    """Per entity, the statement pairs worth showing: one thing stated with different figures.

    Amounts, dates, codes and measurements rarely become nodes, so a conflict of
    that kind exists only in two chunk statements about the entity.
    """
    statements: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for fact in facts.values():
        if is_statement(fact):
            statements[fact["target"]].append((fact["id"], _extract_statement_features(fact)))
    paired: dict[str, list[set[str]]] = {}
    for entity_id, items in statements.items():
        pairs = [
            {first_id, second_id}
            for index, (first_id, first) in enumerate(items)
            for second_id, second in items[index + 1 :]
            if _is_statement_pair(first, second)
        ]
        if pairs:
            paired[entity_id] = pairs
    return paired


# --- what a run compares -----------------------------------------------------


@dataclass(frozen=True)
class FactComparisons:
    """What each entity's facts offer to compare in one run, as ``find_fact_comparisons`` found it."""

    facts_by_endpoint: dict[str, set[str]]  # node id -> the facts touching it
    value_groups: dict[str, list[set[str]]]  # entity id -> its value groups
    statement_pairs: dict[str, list[set[str]]]  # entity id -> its statement pairs


def find_fact_comparisons(
    facts: dict[str, dict], nodes: dict[str, dict], since: datetime | None
) -> FactComparisons:
    """What each entity's facts offer to compare, worked out once per run.

    Per entity: the facts touching it, its value groups and its statement pairs, all
    fixed by the read. On a run with ``since``, an entity with facts observed after it
    keeps only the groups and pairs holding one; an entity with none, such as the
    subject of a repair, keeps them all. Conflict support is not fixed: the writer
    rewrites citations and marks between calls, so ``select_supporting_facts`` stays
    a per-call read.
    """
    facts_by_endpoint: dict[str, set[str]] = defaultdict(set)
    for fact_id, fact in facts.items():
        for endpoint in get_endpoints(fact):
            facts_by_endpoint[endpoint].add(fact_id)
    value_groups = find_value_groups(facts, nodes)
    statement_pairs = find_statement_pairs(facts)
    for entity_id, fact_ids in facts_by_endpoint.items():
        new = {fact_id for fact_id in fact_ids if is_observed_since(facts[fact_id], since)}
        if new:
            for comparisons_by_entity in (value_groups, statement_pairs):
                kept = [
                    comparison
                    for comparison in comparisons_by_entity.pop(entity_id, [])
                    if comparison & new
                ]
                if kept:
                    comparisons_by_entity[entity_id] = kept
    return FactComparisons(dict(facts_by_endpoint), value_groups, statement_pairs)


# --- subjects and their sections ---------------------------------------------


def select_subject_ids(scope: ReviewScope, comparisons: FactComparisons) -> list[str]:
    """The entities worth a call: values or statements to compare, or a stored conflict to judge.

    They keep the reader's order, which decides which subjects share one call.
    """
    conflict_owner_ids = {str(conflict["about_id"]) for conflict in scope.conflicts.values()}
    return [
        entity_id
        for entity_id in scope.entities
        if entity_id in comparisons.value_groups
        or entity_id in comparisons.statement_pairs
        or entity_id in conflict_owner_ids
    ]


def select_section_facts(
    scope: ReviewScope, comparisons: FactComparisons, subject: str
) -> set[str]:
    """A subject's group facts and paired statements, and its facts supporting a stored conflict."""
    return (
        set().union(*comparisons.value_groups.get(subject, []))
        | set().union(*comparisons.statement_pairs.get(subject, []))
        | select_supporting_facts(
            scope.facts,
            comparisons.facts_by_endpoint.get(subject, set()),
            scope.conflicts.values(),
        )
    )


# --- what a section too large for one call may lose --------------------------


def _group_values_by_relationship(facts: dict[str, dict], fact_ids: set[str], subject: str) -> dict:
    """Each relationship name among these facts, with the values it links the subject to.

    A statement names no value, so statements never make a list.
    """
    values: dict[str, set[str]] = defaultdict(set)
    for fact_id in fact_ids:
        fact = facts[fact_id]
        if not is_statement(fact):
            values[fact["relationship"]].update(set(get_endpoints(fact)) - {subject})
    return values


def select_lists_to_leave_out(
    scope: ReviewScope, subject: str, section: set[str]
) -> list[tuple[str, set[str]]]:
    """The list-like names a section may lose to fit one call, in the order to lose them.

    Values a list holds at once are no conflict, and a name with fewer values may be
    one attribute's history, so only lists are left out, most values first. Every fact
    supporting a conflict about the subject stays, and so does every name such a fact
    is marked under: the conflict is judged with every value of its attribute. Each
    name comes with the facts that leaving it out removes.
    """
    facts = scope.facts
    subject_conflicts = {
        conflict_id: conflict
        for conflict_id, conflict in scope.conflicts.items()
        if str(conflict["about_id"]) == subject
    }
    supporting = select_supporting_facts(facts, section, subject_conflicts.values())
    protected_relationship_names = {
        facts[fact_id]["relationship"]
        for fact_id in supporting
        if has_conflict_mark(facts[fact_id], subject_conflicts)
    }
    removable = section - supporting
    lists = {
        name: values
        for name, values in _group_values_by_relationship(facts, removable, subject).items()
        if name not in protected_relationship_names and len(values) >= MIN_LIST_VALUES
    }
    # Most values first; of two names with as many values, the one that sorts first.
    names = sorted(lists, key=lambda name: (-len(lists[name]), name))
    return [
        (name, {fact_id for fact_id in removable if facts[fact_id]["relationship"] == name})
        for name in names
    ]
