"""Which entities a review makes subjects, and which facts each subject's section holds."""

from datetime import datetime, timezone

import pytest

from cognee.tasks.memify.review_conflicts.select_review_facts import (
    find_fact_comparisons,
    find_statement_pairs,
    find_value_groups,
)
from cognee.tests.unit.tasks.memify.review_conflicts.review_builders import (
    alice_group_and_lone_statement,
    build_prompt,
    fact,
)


@pytest.mark.parametrize(
    "case, grouped",
    [
        ("same type", True),
        ("untyped values", True),
        ("two relationship names", True),
        ("opposite directions", True),
        ("two dates under one name", True),
        ("one document", False),
        ("types differ", False),
        ("a date beside an entity", False),
    ],
)
def test_value_groups(scope, case, grouped):
    if case == "untyped values":
        scope.nodes["alice"]["types"] = scope.nodes["bob"]["types"] = []
    elif case == "two relationship names":
        scope.facts["e2"]["relationship"] = "sole_ceo"
    elif case == "opposite directions":
        scope.facts["e2"] = fact("e2", "acme", "2026-01-10", relationship="ceo_of", source="bob")
    elif case == "two dates under one name":
        for fact_id, day in (("e1", "2014-05-12"), ("e2", "2015-05-12")):
            scope.nodes[day] = {"id": day, "name": day, "type": "Timestamp"}
            scope.facts[fact_id] = fact(fact_id, day, None, relationship="incorporated_at")
    elif case == "one document":
        scope.facts["e2"]["sources"][0]["data_id"] = "doc-e1"
    elif case == "types differ":
        scope.nodes["bob"]["types"] = ["Company"]
    elif case == "a date beside an entity":
        scope.nodes["2014"] = {"id": "2014", "name": "2014", "type": "Timestamp"}
        scope.facts["e2"] = fact("e2", "2014", None)
    assert find_value_groups(scope.facts, scope.nodes).get("acme") == (
        [{"e1", "e2"}] if grouped else None
    )


@pytest.mark.parametrize(
    "case, paired",
    [
        ("different figures", True),
        ("same number, different unit", True),
        ("one document", False),
        ("same figures", False),
        ("no figure", False),
        ("nothing else in common", False),
    ],
)
def test_paired_statements(scope, case, paired):
    first, second = "Acme has 120 staff", "Acme has 150 staff"
    if case == "same figures":
        second = first
    elif case == "same number, different unit":
        first, second = "Acme keeps logs for 48 hours", "Acme keeps logs for 48 days"
    elif case == "no figure":
        first, second = "Acme has many staff", "Acme has few staff"
    elif case == "nothing else in common":
        second = "Acme was founded in 1999 by Bob"
    scope.facts = {
        "e1": fact("e1", "acme", "2020-05-01", relationship="contains", text=first),
        "e2": fact("e2", "acme", "2026-01-10", relationship="contains", text=second),
    }
    if case == "one document":
        scope.facts["e2"]["sources"][0]["data_id"] = "doc-e1"
    assert find_statement_pairs(scope.facts).get("acme") == ([{"e1", "e2"}] if paired else None)


def test_watermarked_subject_without_new_facts_shows_its_whole_section(scope):
    # Nothing was written since the last review, so nothing is narrowed.
    scope.since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert build_prompt(scope, ["acme"]).shown_facts["acme"] == ["e2", "e1"]


def test_sections_hold_group_facts_and_paired_statements_only(scope):
    # The claim naming Alice as Acme's CEO belongs to Acme's group, not to one of Alice's,
    # and Acme's lone statement e7 pairs with nothing: neither widens a section.
    alice_group_and_lone_statement(scope)
    prompt = build_prompt(scope, ["acme", "alice"])
    assert prompt.shown_facts == {"acme": ["e2", "e1"], "alice": ["e6", "e5"]}
    assert "[n3] Bob (types: Person)" in prompt.text


def test_reused_comparisons_read_support_changed_by_previous_batch(scope):
    # e3 pairs with nothing, so it is shown only while it supports a stored conflict.
    scope.facts["e3"] = fact("e3", "acme", "2026-09-01", relationship="contains")
    comparisons = find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    assert build_prompt(scope, ["acme"], comparisons).shown_facts["acme"] == ["e2", "e1"]

    # The writer updates the shared input between yielded batches. Neither marks
    # nor citations may be kept with the comparisons, which a run never changes.
    scope.conflicts["saved"] = {
        "id": "saved",
        "about_id": "acme",
        "attribute": "title",
        "sources": ["chunk-e1"],
        "values": [],
    }
    scope.facts["e3"]["properties"]["conflict_marks"] = [
        {"conflict_id": "saved", "status": "current"}
    ]
    assert build_prompt(scope, ["acme"], comparisons).shown_facts["acme"] == [
        "e3",
        "e1",
        "e2",
    ]

    scope.facts["e3"]["properties"]["conflict_marks"].clear()
    scope.conflicts["saved"]["sources"] = ["chunk-e2"]
    assert build_prompt(scope, ["acme"], comparisons).shown_facts["acme"] == ["e2", "e1"]
