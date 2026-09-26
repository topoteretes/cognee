from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.hybrid.chunks import PASSAGES_DROPPED_BY_CUTOFF
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.only_context_prompt import has_context


def _retrieved(*, chunks: list | None = None, dropped: bool = False) -> dict:
    retrieved = {
        "chunks": chunks or [],
        "entities": [{"id": "e1", "name": "Unrelated", "description": "noise entity"}],
        "facts": [{"text": "an unrelated fact that should not be injected"}],
    }
    if dropped:
        retrieved[PASSAGES_DROPPED_BY_CUTOFF] = True
    return retrieved


@pytest.mark.asyncio
async def test_min_score_drops_unscored_channels_before_the_prompt_is_built():
    retriever = HybridRetriever(min_score=0.2)
    context = await retriever.get_context_from_objects(
        query="noise",
        retrieved_objects=_retrieved(dropped=True),
    )

    assert context == ""
    assert not has_context(context)


@pytest.mark.asyncio
async def test_empty_chunk_lane_without_a_cutoff_keeps_entities():
    retriever = HybridRetriever(min_score=0.2)
    context = await retriever.get_context_from_objects(
        query="entities-only",
        retrieved_objects=_retrieved(),
    )

    assert "Unrelated" in context


@pytest.mark.asyncio
async def test_without_min_score_unscored_channels_still_render():
    retriever = HybridRetriever()
    context = await retriever.get_context_from_objects(
        query="noise",
        retrieved_objects=_retrieved(),
    )

    assert "Unrelated" in context
    assert "unrelated fact" in context


@pytest.mark.asyncio
async def test_min_score_keeps_other_channels_when_a_passage_survives():
    retriever = HybridRetriever(min_score=0.2)
    context = await retriever.get_context_from_objects(
        query="relevant",
        retrieved_objects=_retrieved(chunks=[{"id": "c1", "text": "the relevant passage"}]),
    )

    assert "the relevant passage" in context
    assert "Unrelated" in context


@pytest.mark.asyncio
async def test_batch_cutoff_builds_global_context_only_for_survivors(monkeypatch):
    retriever = HybridRetriever(min_score=0.2, include_global_context_index=True)
    calls = []

    async def fake_global(query):
        calls.append(query)
        return f"global:{query}"

    monkeypatch.setattr(retriever, "_build_global_context_section", fake_global)
    contexts = await retriever.get_context_from_objects(
        query_batch=["noise", "keep"],
        retrieved_objects=[
            _retrieved(dropped=True),
            _retrieved(chunks=[{"id": "c1", "text": "kept passage"}]),
        ],
    )

    assert calls == ["keep"]
    assert contexts[0] == ""
    assert "kept passage" in contexts[1]
    assert "global:keep" in contexts[1]


@pytest.mark.asyncio
async def test_retrieve_clears_entities_only_when_cutoff_removes_every_passage(monkeypatch):
    retriever = HybridRetriever(min_score=1.0)
    retriever._unified_engine = SimpleNamespace(
        vector=SimpleNamespace(
            embedding_engine=SimpleNamespace(embed_text=AsyncMock(return_value=[[0.1]]))
        )
    )
    monkeypatch.setattr(
        "cognee.modules.retrieval.hybrid_retriever.build_truth_context",
        AsyncMock(
            return_value=SimpleNamespace(
                q_coords=None, truth_state_by_id={}, current_truth_epoch=None
            )
        ),
    )
    monkeypatch.setattr(
        "cognee.modules.retrieval.hybrid_retriever.load_preference_weights",
        AsyncMock(return_value={}),
    )

    async def dropped_chunks(**kwargs):
        return {"chunks": [], "chunk_summaries": {}, PASSAGES_DROPPED_BY_CUTOFF: True}

    async def empty_without_cutoff(**kwargs):
        return {"chunks": [], "chunk_summaries": {}, PASSAGES_DROPPED_BY_CUTOFF: False}

    async def entities(*args, **kwargs):
        return [{"name": "Noise"}], [{"text": "noise fact"}]

    monkeypatch.setattr(retriever, "_retrieve_entities_and_facts", entities)
    monkeypatch.setattr(
        "cognee.modules.retrieval.hybrid_retriever.retrieve_hybrid_chunks",
        dropped_chunks,
    )
    dropped = await retriever._retrieve_one("noise")
    assert dropped["entities"] == []
    assert dropped["facts"] == []

    monkeypatch.setattr(
        "cognee.modules.retrieval.hybrid_retriever.retrieve_hybrid_chunks",
        empty_without_cutoff,
    )
    kept = await retriever._retrieve_one("entities-only")
    assert kept["entities"] == [{"name": "Noise"}]
    assert kept["facts"] == [{"text": "noise fact"}]
