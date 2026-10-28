"""The prompt one review call sends: headers, fact lines in rank order, labels, documents."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from cognee.tasks.memify.review_conflicts import schema
from cognee.tasks.memify.review_conflicts.accept_review import accept_review_response
from cognee.tasks.memify.review_conflicts.build_review_prompt import build_review_prompt
from cognee.tasks.memify.review_conflicts.select_review_facts import find_fact_comparisons
from cognee.tests.unit.tasks.memify.review_conflicts.review_builders import (
    alice_group_and_lone_statement,
    build_prompt,
    existing_conflict,
    fact,
)


def test_prompt_uses_sources_and_types_without_stored_descriptions(scope):
    scope.entities["acme"]["description"] = "STALE_DESCRIPTION"
    existing_conflict(scope)
    text = build_prompt(scope, ["acme"]).text
    # Fact lines cite documents by label; the last block names each one once, dated.
    assert "| d1 (2026-01-10)\n" in text and text.endswith("[d2] e1.txt")
    assert "Company" in text and "Competing relationships:" in text
    assert "STALE_DESCRIPTION" not in text and "Old explanation" not in text
    assert "[f1] has_ceo" in text


def test_existing_action_labels_belong_only_to_batch_subjects(scope):
    conflict_id = existing_conflict(scope)
    prompt = build_prompt(scope, ["alice"])
    text = prompt.text
    assert "Existing has_ceo (value)" in text
    assert prompt.conflict_labels == {}
    assert "[f1]" not in text
    output = schema.ReviewResponse(reviewed=["n1"], drop=["f1"])
    with pytest.raises(ValueError, match="Unknown, repeated, or replaced drop"):
        accept_review_response(output, scope, prompt)

    owned = build_prompt(scope, ["acme", "alice"])
    assert owned.conflict_labels == {"f1": conflict_id}


def test_legacy_textless_relationship_and_unnamed_sources(scope):
    del scope.facts["e1"]["properties"]["edge_text"]
    scope.facts["e1"]["sources"][0]["document"] = None
    scope.facts["e2"]["properties"]["edge_text"] = None
    scope.facts["e2"]["sources"][0].update(document=None, data_id=None)
    before = deepcopy(scope.facts)
    text = build_prompt(scope, ["acme"]).text
    assert "| has_ceo | d2 (2020-05-01)\n" in text and "[d2] doc-e1" in text
    assert "| has_ceo | d1 (2026-01-10)\n" in text and "[d1] unnamed document\n" in text
    assert scope.facts == before


def test_section_order_follows_priority(scope):
    existing_conflict(scope)
    scope.facts["e2"]["properties"]["conflict_marks"] = []
    scope.conflicts.clear()
    # Two statements with different figures pair up, so both are shown.
    scope.facts["e3"] = fact(
        "e3", "acme", "2026-09-01", relationship="contains", text="Acme has 120 staff"
    )
    scope.facts["e4"] = fact(
        "e4", "acme", "2020-01-01", relationship="contains", text="Acme has 150 staff"
    )
    scope.facts["e4"]["observed_at"] = datetime(2026, 9, 2, tzinfo=timezone.utc)
    # An older pair that holds no new fact.
    for fact_id, date, year in (("e8", "2019-01-01", 1990), ("e9", "2018-01-01", 1991)):
        scope.facts[fact_id] = fact(
            fact_id, "acme", date, relationship="contains", text=f"Acme opened its plant in {year}"
        )
    scope.since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    # Existing conflict support has a citation.
    scope.conflicts["saved"] = {
        "id": "saved",
        "about_id": "acme",
        "attribute": "title",
        "sources": ["chunk-e1"],
        "values": [],
    }
    before = deepcopy(scope.facts)
    prompt = build_prompt(scope, ["acme"])
    # Observed since the last review, then cited, then statements; the group and the
    # older pair hold no new fact, so of them only e1, which the stored conflict cites, is shown.
    assert prompt.shown_facts["acme"] == ["e4", "e1", "e3"]
    assert scope.facts == before
    scope.since = None
    prompt = build_prompt(scope, ["acme"])
    assert prompt.shown_facts["acme"] == ["e1", "e2", "e3", "e4", "e8", "e9"]


def test_competing_line_names_only_labels_in_that_subjects_section(scope):
    # acme's section leaves out e1, which no other section shows, so e1 has no label:
    # acme's group would compare one fact, and gets no line.
    alice_group_and_lone_statement(scope)
    comparisons = find_fact_comparisons(scope.facts, scope.nodes, scope.since)
    sections = {"acme": ["e2", "e7"], "alice": ["e5", "e6"]}
    prompt = build_review_prompt(scope, comparisons, sections)
    acme_section, alice_section = prompt.text.split("\n\n")[:2]
    assert "Competing relationships" not in acme_section
    assert "Competing relationships: r2, r3" in alice_section
