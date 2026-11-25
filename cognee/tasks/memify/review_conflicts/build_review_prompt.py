"""Write one review call's prompt: each subject's section, then the values and documents it names.

One call shows a handful of subjects with the facts of their value groups, their
paired statements, the facts that support a stored conflict, and the conflicts
already stored about them. Everything in a call is labelled, and only labels this
call rendered may come back — ``accept_review_response`` resolves them and rejects
the rest.
"""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime

from cognee.modules.graph.utils.fact_conflicts import effective_date_display

from .facts import (
    NO_DATE,
    get_endpoints,
    is_observed_since,
    is_statement,
    parse_date,
    select_supporting_facts,
)
from .models import ReviewScope
from .select_review_facts import FactComparisons


@dataclass(frozen=True)
class ReviewPrompt:
    """The exact text one call sent, the facts each section showed, and the labels it means.

    Labels belong to one prompt, so each half of a split retry is labelled anew.
    """

    text: str
    shown_facts: dict[str, list[str]]  # each subject, in call order -> its facts, in rank order
    node_labels: dict[str, str]  # `n1` -> node id, for subjects and values
    fact_labels: dict[str, str]  # `r1` or `s1` -> fact id
    conflict_labels: dict[str, str]  # `f1` -> conflict id

    @property
    def subject_ids(self) -> list[str]:
        return list(self.shown_facts)


def build_review_prompt(
    scope: ReviewScope, comparisons: FactComparisons, sections: dict[str, Collection[str]]
) -> ReviewPrompt:
    """The prompt showing these sections, in the order given, each one's facts in rank order."""
    shown_facts = _rank_section_facts(scope, sections)
    subject_ids = list(shown_facts)
    shown = set().union(*shown_facts.values())
    node_labels = _label_nodes(scope, subject_ids, shown)
    fact_labels = _label_facts(scope, shown)
    document_labels = _label_documents(scope, shown_facts)
    conflict_labels = _label_owned_conflicts(scope, subject_ids)
    label_of_node = {node_id: label for label, node_id in node_labels.items()}
    label_of_fact = {fact_id: label for label, fact_id in fact_labels.items()}
    label_of_document = {source.get("data_id"): label for label, source in document_labels.items()}
    label_of_conflict = {conflict_id: label for label, conflict_id in conflict_labels.items()}
    rendered_sections = [
        "\n".join(
            [
                *_render_subject_header(scope, subject, label_of_node[subject], label_of_conflict),
                *(
                    _render_fact_line(
                        scope, fact_id, label_of_node, label_of_fact, label_of_document
                    )
                    for fact_id in shown_facts[subject]
                ),
                *_render_competing_lines(comparisons, subject, shown_facts[subject], label_of_fact),
            ]
        )
        for subject in subject_ids
    ]
    text = "\n\n".join(
        rendered_sections
        + _render_value_names(scope, node_labels, subject_ids)
        + _render_document_block(document_labels)
    )
    return ReviewPrompt(text, shown_facts, node_labels, fact_labels, conflict_labels)


# --- fact order --------------------------------------------------------------


def _rank_fact(
    fact: dict, since: datetime | None, supporting_ids: Collection[str]
) -> tuple[int, float, str]:
    """The sort key that orders a section's fact lines, first line first.

    Written since the last review, then facts supporting a stored conflict, then a
    group claim, then a paired statement; within a tier the most recently dated
    fact comes first.
    """
    if is_observed_since(fact, since):
        tier = 0
    elif fact["id"] in supporting_ids:
        tier = 1
    else:
        tier = 3 if is_statement(fact) else 2
    age = (parse_date(fact.get("effective_date")) - NO_DATE).total_seconds()
    return tier, -age, fact["id"]


def _rank_section_facts(
    scope: ReviewScope, sections: dict[str, Collection[str]]
) -> dict[str, list[str]]:
    """Each section's facts in the order its lines are shown."""
    shown = set().union(*sections.values())
    supporting = select_supporting_facts(scope.facts, shown, scope.conflicts.values())
    sort_keys = {
        fact_id: _rank_fact(scope.facts[fact_id], scope.since, supporting) for fact_id in shown
    }
    return {
        subject: sorted(fact_ids, key=sort_keys.__getitem__)
        for subject, fact_ids in sections.items()
    }


# --- labels ------------------------------------------------------------------


def _assign_labels(prefix: str, items: list) -> dict:
    """`n1`, `n2`, … in the given order: the prompt names nodes, facts and documents by label."""
    return {f"{prefix}{number}": item for number, item in enumerate(items, 1)}


def _label_nodes(scope: ReviewScope, subject_ids: list[str], shown: set[str]) -> dict[str, str]:
    """Subjects keep their call order; the values of shown claims follow, sorted."""
    endpoint_ids = {
        endpoint
        for fact_id in shown
        if not is_statement(scope.facts[fact_id])
        for endpoint in get_endpoints(scope.facts[fact_id])
    }
    return _assign_labels("n", subject_ids + sorted(endpoint_ids - set(subject_ids)))


def _label_facts(scope: ReviewScope, shown: set[str]) -> dict[str, str]:
    """`[s#]` is a chunk statement about a subject, `[r#]` a claim between two entities."""
    statements = sorted(fact_id for fact_id in shown if is_statement(scope.facts[fact_id]))
    relationships = sorted(fact_id for fact_id in shown if not is_statement(scope.facts[fact_id]))
    return {**_assign_labels("s", statements), **_assign_labels("r", relationships)}


def _label_documents(scope: ReviewScope, shown_facts: dict[str, list[str]]) -> dict[str, dict]:
    """`d1`, `d2`, … in order of first citation, each standing for one document."""
    documents: dict[str, dict] = {}
    for fact_ids in shown_facts.values():
        for fact_id in fact_ids:
            for source in scope.facts[fact_id]["sources"]:
                documents.setdefault(source.get("data_id"), source)
    return _assign_labels("d", list(documents.values()))


def _label_owned_conflicts(scope: ReviewScope, subject_ids: list[str]) -> dict[str, str]:
    """`f1`, `f2`, … for the stored conflicts this call's subjects own, for the response to cite.

    Only the subject that owns a conflict may replace or drop it, so no other conflict
    gets a label. Sorted by id: the stored order follows adapter rows.
    """
    subjects = set(subject_ids)
    return _assign_labels(
        "f",
        [
            conflict_id
            for conflict_id, conflict in sorted(scope.conflicts.items())
            if str(conflict["about_id"]) in subjects
        ],
    )


# --- rendering ---------------------------------------------------------------


def _render_subject_header(
    scope: ReviewScope, subject: str, label: str, label_of_conflict: dict[str, str]
) -> list[str]:
    """Name the subject with its types, then each stored conflict it is the subject or a value of."""
    properties = scope.entities[subject]
    types = ", ".join(properties.get("types", []))
    lines = [f"Subject [{label}] {properties['name']} (types: {types})"]
    # In id order, the order of their `f#` labels.
    for conflict_id, conflict in sorted(scope.conflicts.items()):
        if subject == str(conflict["about_id"]):
            role = "subject"
        elif subject in conflict.get("values", []):
            role = "value"
        else:
            continue
        cited = f" [{label_of_conflict[conflict_id]}]" if conflict_id in label_of_conflict else ""
        pending = "; pending write" if role == "subject" and conflict.get("write_pending") else ""
        lines.append(f"Existing{cited} {conflict['attribute']} ({role}{pending})")
    return lines


def _render_citation(label: str, source: dict) -> str:
    """A cited document's label and, where known, the date it states its facts for."""
    date = effective_date_display(source.get("effective_date"))
    return f"{label} ({date})" if date else label


def _render_fact_line(
    scope: ReviewScope,
    fact_id: str,
    label_of_node: dict[str, str],
    label_of_fact: dict[str, str],
    label_of_document: dict[str, str],
) -> str:
    fact = scope.facts[fact_id]
    relation = (
        "statement"
        if is_statement(fact)
        else f"{label_of_node[fact['source']]} --{fact['relationship']}--> "
        f"{label_of_node[fact['target']]}"
    )
    statement = fact["properties"].get("edge_text") or fact["relationship"]
    sources = ", ".join(
        dict.fromkeys(
            _render_citation(label_of_document[source.get("data_id")], source)
            for source in fact["sources"]
        )
    )
    return f"[{label_of_fact[fact_id]}] {relation} | {statement} | {sources}"


def _render_competing_lines(
    comparisons: FactComparisons,
    subject: str,
    fact_ids: list[str],
    label_of_fact: dict[str, str],
) -> list[str]:
    """One line per value group of this subject, naming the labels its section shows."""
    section = set(fact_ids)
    lines = []
    for group in comparisons.value_groups.get(subject, []):
        # Only facts this section shows: a group fact it left out, such as one lost with
        # a list, may have no label in this call at all.
        labels = sorted(label_of_fact[fact_id] for fact_id in group & section)
        if len(labels) > 1:
            lines.append(f"Competing relationships: {', '.join(labels)}")
    return lines


def _render_value_names(
    scope: ReviewScope, node_labels: dict[str, str], subject_ids: list[str]
) -> list[str]:
    """Values are named once, after the sections, with their types."""
    subjects = set(subject_ids)
    lines = []
    for label, node_id in node_labels.items():
        if node_id in subjects:
            continue
        node = scope.nodes.get(node_id, {})
        types = ", ".join(node.get("types", []))
        lines.append(
            f"[{label}] {node.get('name', node_id)}" + (f" (types: {types})" if types else "")
        )
    return lines


def _render_document_block(documents: dict[str, dict]) -> list[str]:
    """Every cited document once, `[d#] name`, as the call's last block."""
    lines = [
        f"[{label}] {source.get('document') or source.get('data_id') or 'unnamed document'}"
        for label, source in documents.items()
    ]
    return ["\n".join(lines)] if lines else []
