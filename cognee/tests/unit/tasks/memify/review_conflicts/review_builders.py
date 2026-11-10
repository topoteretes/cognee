"""Facts, stored conflicts, prompts and answers the reviewer's tests share."""

from cognee.modules.engine.models.FactConflict import FactConflict
from cognee.tasks.memify.review_conflicts import schema
from cognee.tasks.memify.review_conflicts.build_review_prompt import build_review_prompt
from cognee.tasks.memify.review_conflicts.select_review_facts import (
    find_fact_comparisons,
    select_section_facts,
)


def fact(fact_id, target, date, *, relationship="has_ceo", text=None, source=None):
    return {
        "id": fact_id,
        "source": source or ("acme" if relationship != "contains" else f"chunk-{fact_id}"),
        "target": target,
        "relationship": relationship,
        "properties": {"edge_text": text or f"{target} is Acme's CEO", "conflict_marks": []},
        "sources": [
            {
                "chunk_id": f"chunk-{fact_id}",
                "data_id": f"doc-{fact_id}",
                "document": f"{fact_id}.txt",
                "effective_date": date,
            }
        ],
        "effective_date": date,
        "observed_at": None,
    }


def alice_group_and_lone_statement(scope):
    """Alice's languages form a group of her own; Acme's lone statement e7 pairs with nothing."""
    scope.facts["e5"] = fact("e5", "english", "2026-06-01", relationship="speaks", source="alice")
    scope.facts["e6"] = fact("e6", "german", "2026-06-02", relationship="speaks", source="alice")
    scope.facts["e7"] = fact("e7", "acme", "2019-01-01", relationship="contains")


def build_prompt(scope, subjects, comparisons=None, *, leaving_out=()):
    """The prompt for these subjects, each section selected as the planner selects it.

    ``leaving_out`` names facts to take out of every section, for an answer checked
    against a prompt that did not show them.
    """
    comparisons = comparisons or find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    sections = {
        subject: select_section_facts(scope, comparisons, subject) - set(leaving_out)
        for subject in subjects
    }
    return build_review_prompt(scope, comparisons, sections)


def response(**conflict):
    return schema.ReviewResponse(
        reviewed=["n1", "n2", "n3"],
        conflicts=[
            schema.ReportedConflict(
                about="n1",
                attribute="ceo",
                kind="time_varying",
                text="Bob succeeded Alice.",
                current=["r2"],
                superseded=["r1"],
                **conflict,
            )
        ],
    )


def existing_conflict(scope):
    conflict = FactConflict(
        dataset_id="dataset",
        about_id="acme",
        attribute="has_ceo",
        kind="time_varying",
        status="resolved",
        text="Old explanation",
        values=["alice", "bob"],
        sources=["chunk-e1", "chunk-e2"],
    )
    scope.conflicts[str(conflict.id)] = conflict.model_dump(mode="json")
    for value in scope.facts.values():
        value["properties"]["conflict_marks"] = [
            {"conflict_id": str(conflict.id), "status": "current"}
        ]
    return str(conflict.id)
