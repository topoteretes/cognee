from unittest.mock import MagicMock

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid.facts import (
    connection_edge_type_id,
    edge_rank_by_id,
    format_facts,
    graph_evidence_by_edge_type_id,
    select_facts,
)


def _edge_id(text):
    return str(EdgeType.id_for(text))


def _hit(text):
    hit = MagicMock()
    hit.id = _edge_id(text)
    hit.payload = {"text": text}
    return hit


def test_connection_edge_type_id_prefers_top_level_edge_text():
    edge = {"edge_text": "Alice works at Acme.", "relationship_name": "works_at"}

    assert connection_edge_type_id(edge) == _edge_id("Alice works at Acme.")


def test_connection_edge_type_id_reads_nested_properties_edge_text():
    edge = {"properties": {"edge_text": "Alice works at Acme."}, "relationship_name": "works_at"}

    assert connection_edge_type_id(edge) == _edge_id("Alice works at Acme.")


def test_connection_edge_type_id_falls_back_to_relationship_name():
    assert connection_edge_type_id({"relationship_name": "works_at"}) == _edge_id("works_at")


def test_connection_edge_type_id_returns_none_without_any_text():
    assert connection_edge_type_id({}) is None
    assert connection_edge_type_id({"edge_text": "  ", "relationship_name": ""}) is None


def test_edge_rank_by_id_keeps_first_occurrence_of_duplicate_ids():
    hits = [
        _hit("Alice works at Acme."),
        _hit("Bob founded Initech."),
        _hit("Alice works at Acme."),
    ]

    assert edge_rank_by_id(hits) == {
        _edge_id("Alice works at Acme."): 0,
        _edge_id("Bob founded Initech."): 1,
    }


def test_select_facts_respects_hit_order_and_top_k():
    hits = [
        _hit("Alice works at Acme."),
        _hit("Bob founded Initech."),
        _hit("Carol leads the data team."),
    ]

    facts = select_facts(hits, set(), 2)

    assert [fact["text"] for fact in facts] == ["Alice works at Acme.", "Bob founded Initech."]


def test_select_facts_skips_excluded_short_and_invalid_hits():
    shown_as_bullet = _hit("Alice works at Acme.")
    aggregate_row = _hit("works at")
    textless = _hit("Bob founded Initech.")
    textless.payload = {}
    idless = _hit("Carol leads the data team.")
    idless.id = None
    kept = _hit("Dora reviews proposals weekly.")

    facts = select_facts(
        [shown_as_bullet, aggregate_row, textless, idless, kept],
        {shown_as_bullet.id},
        5,
    )

    assert [fact["text"] for fact in facts] == ["Dora reviews proposals weekly."]


def test_select_facts_falls_back_to_relationship_name_payload():
    hit = _hit("ignored")
    hit.payload = {"relationship_name": "Alice works at Acme."}

    facts = select_facts([hit], set(), 5)

    assert [fact["text"] for fact in facts] == ["Alice works at Acme."]


def test_select_facts_rewrites_contains_edge_texts_as_glossary_entries():
    hit = _hit("Document chunk mentions frostline: Project that tracks temperature risk.")

    facts = select_facts([hit], set(), 5)

    assert [fact["text"] for fact in facts] == ["Frostline: Project that tracks temperature risk."]
    assert facts[0]["id"] == hit.id


def test_select_facts_returns_empty_for_empty_hits_or_zero_top_k():
    assert select_facts([], set(), 5) == []
    assert select_facts([_hit("Alice works at Acme.")], set(), 0) == []


def test_select_facts_can_require_resolvable_graph_evidence():
    hit = _hit("Alice works at Acme.")
    evidence = {
        hit.id: [
            {
                "source": "Alice",
                "source_id": "alice",
                "relationship": "works_at",
                "target": "Acme",
                "target_id": "acme",
                "provenance": {"source_chunk_id": "chunk-1"},
            }
        ]
    }

    assert select_facts([hit], set(), 5, require_evidence=True) == []
    assert select_facts([hit], set(), 5, evidence_by_id=evidence, require_evidence=True) == [
        {
            "id": hit.id,
            "text": "Alice works at Acme.",
            "evidence": evidence[hit.id],
        }
    ]


def test_graph_evidence_is_deduplicated_and_keeps_temporal_provenance():
    edge_id = _edge_id("Alice works at Acme.")
    edge = {
        "edge_type_id": edge_id,
        "source": "Alice",
        "source_id": "alice",
        "relationship": "works_at",
        "target": "Acme",
        "target_id": "acme",
        "provenance": {"valid_from": "2026-03-01"},
    }

    evidence = graph_evidence_by_edge_type_id([{"edges": [edge, dict(edge)]}])

    assert evidence[edge_id] == [
        {
            "source": "Alice",
            "source_id": "alice",
            "relationship": "works_at",
            "target": "Acme",
            "target_id": "acme",
            "provenance": {"valid_from": "2026-03-01"},
        }
    ]


def test_select_facts_unicode_normalizes_deduplication():
    composed = _hit("Caf\u00e9 has a multilingual menu.")
    decomposed = _hit("Cafe\u0301   has a multilingual menu.")

    facts = select_facts([composed, decomposed], set(), 5)

    assert [fact["text"] for fact in facts] == ["Caf\u00e9 has a multilingual menu."]


def test_select_facts_accepts_substantive_cjk_text_without_spaces():
    fact = _hit(
        "\u30a2\u30ea\u30b9\u306f\u30a2\u30af\u30e1\u793e\u3067\u30c7\u30fc\u30bf\u30c1\u30fc\u30e0\u3092\u7387\u3044\u3066\u3044\u308b"
    )

    facts = select_facts([fact], set(), 5)

    assert [item["text"] for item in facts] == [fact.payload["text"]]


def test_format_facts():
    facts = [
        {"id": "fact-1", "text": "Alice works at Acme."},
        {"id": "fact-2", "text": "Bob founded Initech."},
    ]

    assert format_facts([]) == ""
    assert format_facts(facts) == (
        "## Related facts\n- Alice works at Acme.\n- Bob founded Initech."
    )


def test_format_facts_renders_graph_evidence():
    facts = [
        {
            "id": "fact-1",
            "text": "Alice works at Acme.",
            "evidence": [
                {
                    "source": "Alice",
                    "relationship": "works_at",
                    "target": "Acme",
                    "provenance": {
                        "source_chunk_id": "chunk-1",
                        "page": 3,
                        "valid_from": "2026-03-01",
                    },
                }
            ],
        }
    ]

    assert format_facts(facts) == (
        "## Related facts\n"
        "- Alice works at Acme. [graph evidence: Alice -- works_at -- Acme; "
        "source: chunk-1, page 3; valid: 2026-03-01 to open]"
    )
