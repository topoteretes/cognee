from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid.context import (
    format_hybrid_context,
    format_hybrid_context_batch,
)
from cognee.modules.retrieval.hybrid.entities import build_entities
from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results
from cognee.modules.retrieval.utils import conflict_context
from cognee.modules.retrieval.utils.conflict_context import ChunkConflicts


@pytest.fixture
def graph():
    graph = AsyncMock()
    graph.get_neighborhood.return_value = (
        [
            (
                "acme",
                {
                    "name": "Acme",
                    "description": "Bob leads\n Acme now.",
                    "conflicts_reviewed_at": "2026-01-10",
                    "belongs_to_set": ["team"],
                },
            ),
            ("company", {"name": "company", "type": "EntityType"}),
            ("alice", {"name": "Alice", "belongs_to_set": ["team"]}),
            ("bob", {"name": "Bob", "belongs_to_set": ["team"]}),
            (
                "f1",
                {
                    "type": "FactConflict",
                    "text": "Bob succeeded Alice.",
                    "belongs_to_set": ["team"],
                },
            ),
        ],
        [
            ("f1", "acme", "conflict_about", {"edge_text": "Bob succeeded Alice."}),
            ("acme", "company", "is_a", {}),
            (
                "acme",
                "alice",
                "has_ceo",
                {
                    "edge_text": "Alice leads Acme.",
                    "conflict_marks": [{"status": "superseded"}],
                    "effective_date": "2020-05-01",
                },
            ),
            (
                "acme",
                "bob",
                "has_ceo",
                {
                    "edge_text": "Bob leads Acme.",
                    "conflict_marks_json": '[{"status":"current"}]',
                    "effective_date": "2026-01-10",
                },
            ),
        ],
    )
    return graph


@pytest.mark.asyncio
async def test_hybrid_hydrates_reviewed_description_and_types_before_rendering(graph):
    entities, _ = await build_entities(
        graph, [{"id": "acme", "text": "Acme", "type": "IndexSchema"}], 3
    )
    entity = entities[0]
    assert entity["type"] == "company"
    assert entity["description"] == "Bob leads Acme now."
    assert entity["conflicts"] == ["Bob succeeded Alice."]
    assert [edge["text"] for edge in entity["edges"]] == [
        "Acme -- is_a -- company",
        "Bob leads Acme. [as of 2026-01-10]",
        "Alice leads Acme. [superseded; as of 2020-05-01]",
    ]
    context = format_hybrid_context("", {"entities": entities})
    assert "### Acme (company)\nBob leads Acme now." in context
    assert context.endswith("## Fact conflicts\n- Bob succeeded Alice.")


@pytest.mark.asyncio
async def test_rank_cut_retains_a_relevant_superseded_fact(graph):
    entities, _ = await build_entities(
        graph,
        [{"id": "acme", "name": "Acme"}],
        2,
        edge_ranks={str(EdgeType.id_for("Alice leads Acme.")): 0},
    )
    bullets = [edge["text"] for edge in entities[0]["edges"]]
    assert len(bullets) == 2
    assert bullets[-1].startswith("Alice leads Acme. [superseded;")
    assert not any(text.startswith("Bob leads Acme.") for text in bullets)


@pytest.mark.asyncio
async def test_nodeset_scope_hides_entity_description_and_entity_reached_conflicts(graph):
    entities, _ = await build_entities(
        graph, [{"id": "acme", "name": "Acme"}], 5, node_name=["team"]
    )
    assert entities[0]["description"] is None
    assert not entities[0].get("conflicts")
    assert entities[0]["type"] == "company"


@pytest.mark.asyncio
async def test_unreviewed_description_is_not_presented(graph):
    graph.get_neighborhood.return_value[0][0][1]["conflicts_reviewed_at"] = None
    entities, _ = await build_entities(
        graph, [{"id": "acme", "name": "Acme", "description": "old vector value"}], 0
    )
    assert entities[0]["description"] is None


@pytest.mark.asyncio
async def test_scoped_fallback_cannot_expose_a_description_from_the_vector_payload(graph):
    graph.get_neighborhood.side_effect = RuntimeError("offline")
    entities, _ = await build_entities(
        graph,
        [{"id": "acme", "name": "Acme", "description": "unscoped description"}],
        0,
        node_name=["team"],
    )
    assert entities[0]["description"] is None


@pytest.mark.asyncio
async def test_chunk_attachment_supports_results_dicts_tuples_and_summary_sources(monkeypatch):
    lookup = AsyncMock(
        return_value={
            "c1": ChunkConflicts("Report", "2026-01-10T09:00:00Z", {"f2": "Second", "f1": "First"})
        }
    )
    monkeypatch.setattr(conflict_context, "get_chunk_conflicts", lookup)
    scored = SimpleNamespace(id="c1", payload={"text": "scored"})
    plain = {"id": "c1", "text": "plain", "source": "original", "_custom": "preserved"}
    pair = ({"id": "c1", "text": "pair"}, 0.2)
    await conflict_context.attach_chunk_conflicts([scored, plain, pair], graph_engine=graph)
    lookup.assert_awaited_once_with(graph, ["c1"])
    assert (
        scored.payload["conflicts"]
        == plain["conflicts"]
        == pair[0]["conflicts"]
        == ["First", "Second"]
    )
    assert plain["_passage_header"] == "Report (2026-01-10)"
    assert conflict_context.public_chunk_payload(plain) == {
        "id": "c1",
        "text": "plain",
        "source": "original",
        "_custom": "preserved",
        "conflicts": ["First", "Second"],
    }
    assert conflict_context.join_passage_texts([pair]) == "source: Report (2026-01-10)\npair"

    summary = {"id": "summary", "source_chunk_id": "c1", "text": "summary"}
    orphan = {"id": "orphan", "text": "no source"}
    await conflict_context.attach_chunk_conflicts(
        [summary, orphan], graph_engine=graph, summaries=True
    )
    assert summary["conflicts"] == ["First", "Second"]
    assert "conflicts" not in orphan
    lookup.assert_awaited_with(graph, ["c1"])


def test_context_deduplicates_conflicts_and_preserves_metadata_and_related_only_facts():
    objects = {
        "chunks": [
            {
                "id": "c1",
                "text": "Passage",
                "_passage_header": "Report (2026-01-10)",
                "external_metadata": {"team": "A"},
                "conflicts": ["Shared dispute"],
            }
        ],
        "entities": [{"name": "Acme", "conflicts": ["Shared dispute", "Another dispute"]}],
        "facts": [{"text": "Shared dispute"}, {"text": "Related-only dispute"}],
    }
    context = format_hybrid_context("", objects)
    assert "source: Report (2026-01-10)\nteam: A\nPassage" in context
    assert "## Related facts\n- Related-only dispute" in context
    assert context.count("Shared dispute") == 1
    assert context.endswith("## Fact conflicts\n- Another dispute\n- Shared dispute")
    assert format_hybrid_context_batch(["", ""], [objects, {}]) == [context, ""]


def test_session_merge_only_renders_conflicts_of_surviving_chunks():
    merged = merge_hybrid_results(
        {
            "chunks": [
                {
                    "id": "keep",
                    "text": "Retained",
                    "conflicts": ["Keep dispute"],
                    "_passage_header": "Kept report",
                }
            ]
        },
        {
            "chunks": [
                {
                    "id": "drop",
                    "text": "Discarded",
                    "conflicts": ["Drop dispute"],
                    "_passage_header": "Discarded report",
                }
            ]
        },
        chunks_limit=1,
        entities_limit=0,
        facts_limit=0,
    )
    context = format_hybrid_context("", merged)
    assert "Keep dispute" in context
    assert "Drop dispute" not in context
    assert "source: Kept report\nRetained" in context
    assert "Discarded report" not in context


def test_plain_passages_keep_their_previous_whitespace():
    assert (
        conflict_context.join_passage_texts([{"text": " First "}, {"text": "Second"}])
        == " First \nSecond"
    )
    assert conflict_context.passage_header("Report", None) == "Report"
    assert conflict_context.format_conflicts([]) == ""
