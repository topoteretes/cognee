"""The retrieval pieces recall's cited mode and expand build on (SDK-1099)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.retrieval import hybrid_retriever as hybrid_retriever_module
from cognee.modules.retrieval import only_context_prompt
from cognee.modules.retrieval.hybrid.chunks import chunk_scores_by_id
from cognee.modules.retrieval.hybrid.context import format_passages, render_passage
from cognee.modules.retrieval.hybrid.entities import (
    build_entities,
    format_entities,
    render_entity,
    render_full_entity,
    stamp_full_version,
)
from cognee.modules.retrieval.hybrid.pairs import chunk_summary_pairs
from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils import resolved_context
from cognee.modules.retrieval.utils.completion import SessionPrompt


def _hit(text: str, score: float, **fields) -> ScoredResult:
    point_id = uuid4()
    return ScoredResult(
        id=point_id, score=score, payload={"id": str(point_id), "text": text, **fields}
    )


def test_section_renderers_join_the_single_record_renderers():
    chunks = [
        {"text": "First passage."},
        {"text": "", "id": "empty"},
        {"text": "Second passage.", "external_metadata": {"team": "sdk"}},
    ]
    assert format_passages(chunks) == "## Relevant passages\n" + "\n---\n".join(
        body for body in map(render_passage, chunks) if body
    )
    entities = [
        {
            "id": "e1",
            "name": "tenant server",
            "type": "software",
            "edges": [{"text": "runs cognee"}],
        },
        {"id": "e2", "name": "plugin", "type": None, "edges": []},
    ]
    assert format_entities(entities) == "## Relevant entities\n" + "\n\n".join(
        render_entity(entity) for entity in entities
    )


def test_chunk_scores_keep_the_fused_score_and_the_closer_raw_distance():
    chunk = _hit("Chunk text", 0.4)
    summary = _hit("Summary", 0.1, source_chunk_id=chunk.payload["id"])
    pairs = chunk_summary_pairs([chunk], [summary])
    ranked = rank_chunk_summary_pairs(pairs, 5, use_importance_weight=False)

    scores = chunk_scores_by_id(ranked)[chunk.payload["id"]]
    assert scores["distance"] == pytest.approx(0.1)
    assert scores["score"] > 0


@pytest.mark.asyncio
async def test_only_context_prompt_keeps_its_frame_and_still_unpacks(monkeypatch):
    monkeypatch.setattr(
        only_context_prompt,
        "load_read_only_session_prompt",
        AsyncMock(return_value=SessionPrompt(history="Earlier turns", guidance="Rules")),
    )
    retriever = SimpleNamespace(
        user_prompt_path="hybrid_context_for_question.txt",
        system_prompt_path="answer_simple_question.txt",
        system_prompt=None,
        session_id=None,
    )
    prompt = await only_context_prompt.build_only_context_prompt(
        retriever, query="why?", context="## Relevant passages\nctx"
    )

    user_prompt, system_prompt = prompt
    frame = prompt.frame
    assert frame.history == "Earlier turns" and frame.guidance == "Rules"
    rebuilt = "\n\n".join(
        [
            frame.history,
            frame.template_prefix + "## Relevant passages\nctx" + frame.template_suffix,
            frame.guidance,
        ]
    )
    assert rebuilt == user_prompt
    assert system_prompt


@pytest.mark.asyncio
async def test_hybrid_expand_reads_records_by_id_and_renders_them(monkeypatch):
    passage = _hit("Full passage body", 0.0, document_id=str(uuid4()))
    fact = ScoredResult(
        id=uuid4(), score=0.0, payload={"text": "Pinning keeps recall on the active dataset."}
    )
    stored = {"DocumentChunk_text": [passage], "EdgeType_relationship_name": [fact]}

    async def retrieve(collection, ids):
        return [point for point in stored.get(collection, []) if str(point.id) in ids]

    engine = SimpleNamespace(vector=SimpleNamespace(retrieve=retrieve), graph=SimpleNamespace())
    monkeypatch.setattr(
        hybrid_retriever_module, "get_unified_engine", AsyncMock(return_value=engine)
    )
    monkeypatch.setattr(hybrid_retriever_module, "dlt_rows_indexed", AsyncMock(return_value=False))
    monkeypatch.setattr(hybrid_retriever_module, "attach_chunk_conflicts", AsyncMock())
    monkeypatch.setattr(resolved_context, "read_user_resolutions", AsyncMock(return_value=[]))

    found = await HybridRetriever().expand_records(
        {
            "passage": [passage.payload["id"], "gone"],
            "fact": [str(fact.id)],
        },
    )

    assert found[("passage", passage.payload["id"])]["body"] == "Full passage body"
    assert found[("fact", str(fact.id))]["body"] == "Pinning keeps recall on the active dataset."
    assert ("passage", "gone") not in found


def _neighborhood(entity_id: str, targets: list[str]):
    """A one-hop read: the entity, its neighbours and an edge to each."""
    nodes = [(entity_id, {"name": "tenant server", "type": "Entity"})] + [
        (str(index), {"name": name}) for index, name in enumerate(targets)
    ]
    edges = [(entity_id, str(index), "runs", {}) for index in range(len(targets))]
    return nodes, edges


@pytest.mark.asyncio
async def test_an_entity_keeps_every_edge_beside_its_ranked_bullets():
    entity_hit = _hit("tenant server", 0.2, name="tenant server")
    entity_id = entity_hit.payload["id"]
    graph = SimpleNamespace(
        get_neighborhood=AsyncMock(return_value=_neighborhood(entity_id, ["zeta", "alpha"]))
    )

    (entity,), _ = await build_entities(graph, [entity_hit], max_edges_per_entity=1)

    assert len(entity["edges"]) == 1
    # Every edge, sorted by text, whatever the cap and the query ranking kept.
    assert [edge["text"] for edge in entity["all_edges"]] == [
        "tenant server -- runs -- alpha",
        "tenant server -- runs -- zeta",
    ]
    assert render_full_entity(entity) == (
        "### tenant server\n- tenant server -- runs -- alpha\n- tenant server -- runs -- zeta"
    )


@pytest.mark.asyncio
async def test_the_full_body_leaves_the_retriever_as_its_hash():
    entity_hit = _hit("tenant server", 0.2, name="tenant server")
    entity_id = entity_hit.payload["id"]
    nodes, edges = _neighborhood(entity_id, ["zeta", "alpha"])
    rejected = str(uuid4())
    # The edge to "zeta" carries the id a user resolution rejects.
    edges[0] = (*edges[0][:3], {"edge_object_id": rejected})
    graph = SimpleNamespace(get_neighborhood=AsyncMock(return_value=(nodes, edges)))
    (entity,), _ = await build_entities(graph, [entity_hit], max_edges_per_entity=1)
    # A user resolution rejected one edge: it is in neither body.
    resolved_context.apply_entity_resolutions(
        [entity],
        [{"about_id": entity_id, "selected_fact": "kept", "rejected_fact_ids": [rejected]}],
    )
    full_body = render_full_entity(entity)
    assert full_body == "### tenant server\n- tenant server -- runs -- alpha"

    stamp_full_version(entity, citable=True)

    # What expand returns for this entity hashes to the cited version; the edge list
    # itself is gone, so a search result does not grow with the neighborhood.
    assert "all_edges" not in entity
    assert entity["full_version"] == compute_content_hash(full_body)
    assert entity["full_chars"] == len(full_body)


@pytest.mark.asyncio
async def test_an_entity_expand_cannot_rebuild_gets_no_version():
    entity_hit = _hit("tenant server", 0.2, name="tenant server")
    entity_id = entity_hit.payload["id"]
    graph = SimpleNamespace(
        get_neighborhood=AsyncMock(return_value=_neighborhood(entity_id, ["alpha"]))
    )
    # An ontology-pinned concept: its heading and edge read are the grounding's own.
    (pinned,), _ = await build_entities(graph, [entity_hit], max_edges_per_entity=1)
    stamp_full_version(pinned, citable=False)
    # A failed neighborhood read: no edges to hash.
    graph.get_neighborhood = AsyncMock(side_effect=RuntimeError("graph down"))
    (unread,), _ = await build_entities(graph, [entity_hit], max_edges_per_entity=1)
    stamp_full_version(unread, citable=True)

    for entity in (pinned, unread):
        assert not {"all_edges", "full_version", "full_chars"} & set(entity)


def _entity_store(monkeypatch, entity_hit, graph):
    async def retrieve(collection, ids):
        return (
            [entity_hit] if collection == "Entity_name" and entity_hit.payload["id"] in ids else []
        )

    embed = AsyncMock()
    engine = SimpleNamespace(
        vector=SimpleNamespace(
            retrieve=retrieve, embedding_engine=SimpleNamespace(embed_text=embed)
        ),
        graph=graph,
    )
    monkeypatch.setattr(
        hybrid_retriever_module, "get_unified_engine", AsyncMock(return_value=engine)
    )
    monkeypatch.setattr(hybrid_retriever_module, "dlt_rows_indexed", AsyncMock(return_value=False))
    monkeypatch.setattr(resolved_context, "read_user_resolutions", AsyncMock(return_value=[]))
    return embed


@pytest.mark.asyncio
async def test_hybrid_expand_returns_an_entity_with_every_edge_and_no_query(monkeypatch):
    entity_hit = _hit("tenant server", 0.0, name="tenant server")
    entity_id = entity_hit.payload["id"]
    # The adapter returns edges in any order; the body does not depend on it.
    reads = [
        _neighborhood(entity_id, ["zeta", "alpha"]),
        _neighborhood(entity_id, ["alpha", "zeta"]),
    ]
    graph = SimpleNamespace(get_neighborhood=AsyncMock(side_effect=reads))
    embed = _entity_store(monkeypatch, entity_hit, graph)

    first = await HybridRetriever(max_edges_per_entity=1).expand_records({"entity": [entity_id]})
    second = await HybridRetriever(max_edges_per_entity=5).expand_records({"entity": [entity_id]})

    body = first[("entity", entity_id)]["body"]
    assert body == (
        "### tenant server\n- tenant server -- runs -- alpha\n- tenant server -- runs -- zeta"
    )
    assert second[("entity", entity_id)]["body"] == body
    embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_entity_with_two_types_has_one_version_whatever_the_edge_order(monkeypatch):
    entity_hit = _hit("apple", 0.0, name="apple")
    entity_id = entity_hit.payload["id"]

    def typed(type_names: list[str]):
        nodes = [(entity_id, {"name": "apple", "type": "Entity"})] + [
            (f"type-{name}", {"name": name}) for name in type_names
        ]
        return nodes, [(entity_id, f"type-{name}", "is_a", {}) for name in type_names]

    # The search and the expand read the same two is_a edges in opposite orders.
    reads = [typed(["company", "fruit"]), typed(["fruit", "company"])]
    graph = SimpleNamespace(get_neighborhood=AsyncMock(side_effect=reads))
    _entity_store(monkeypatch, entity_hit, graph)

    (searched,), _ = await build_entities(graph, [entity_hit], max_edges_per_entity=5)
    stamp_full_version(searched, citable=True)
    expanded = await HybridRetriever().expand_records({"entity": [entity_id]})

    body = expanded[("entity", entity_id)]["body"]
    assert body.startswith("### apple (company)\n")
    assert searched["full_version"] == compute_content_hash(body)


@pytest.mark.asyncio
async def test_hybrid_expand_fails_when_the_graph_read_fails(monkeypatch):
    # build_entities would return the entity without edges: a new version, as if it
    # had changed.
    entity_hit = _hit("tenant server", 0.0, name="tenant server")
    graph = SimpleNamespace(get_neighborhood=AsyncMock(side_effect=RuntimeError("graph down")))
    _entity_store(monkeypatch, entity_hit, graph)

    with pytest.raises(RuntimeError, match="graph down"):
        await HybridRetriever().expand_records({"entity": [entity_hit.payload["id"]]})
