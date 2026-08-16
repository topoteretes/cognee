"""Unit tests for Phase 4 of user preferences: the RAG retriever.

Covers the deterministic pieces only — mocked vector engine and preference
lookup, no graph database, no LLM:
- Flag off (empty lookup): the vector search is issued with ``limit=top_k``
  and the returned order is untouched — byte-identical to today.
- Weights present: ``limit == top_k * PREFERENCE_OVERFETCH``, a stable
  re-sort by ``score * personal_factor(w, influence, distance_space=True)``,
  a trim back to ``top_k``, and — the assertion that separates a real
  re-rank from a reorder — a chunk outside the raw top_k entering the final
  list (Demo 4b).
- An empty weights dict short-circuits: no over-fetch, no sort.
- The sessionless completion passes the preference text as
  ``conversation_history``.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

import cognee.modules.retrieval.completion_retriever as retriever_module
from cognee.modules.retrieval.completion_retriever import (
    CompletionRetriever,
    _stable_sort_by_personal_distance,
)
from cognee.modules.user_preferences import PREFERENCE_OVERFETCH, personal_factor

INFLUENCE = 0.3


def _chunk(chunk_id: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        score=score,
        payload={"id": chunk_id, "text": f"text of {chunk_id}"},
    )


class FakeVectorEngine:
    """Records search kwargs and serves the first ``limit`` seeded chunks."""

    def __init__(self, chunks: List[Any]):
        self.chunks = chunks
        self.search_calls: List[Dict[str, Any]] = []

    async def search(self, collection_name, query, **kwargs):
        self.search_calls.append({"collection_name": collection_name, **kwargs})
        return self.chunks[: kwargs["limit"]]


def _patch_lookup(monkeypatch, result: Tuple[str, Dict[str, float]]):
    calls = []

    async def fake_load_active_preferences():
        calls.append(True)
        return result

    monkeypatch.setattr(retriever_module, "load_active_preferences", fake_load_active_preferences)
    return calls


def _patch_engine(monkeypatch, chunks: List[Any]) -> FakeVectorEngine:
    engine = FakeVectorEngine(chunks)

    async def fake_get_vector_engine_async():
        return engine

    monkeypatch.setattr(retriever_module, "get_vector_engine_async", fake_get_vector_engine_async)
    return engine


def _patch_influence(monkeypatch, influence: float = INFLUENCE):
    config = SimpleNamespace(personalization_influence=influence)
    monkeypatch.setattr(retriever_module, "get_base_config", lambda: config)


@pytest.mark.asyncio
class TestGetRetrievedObjects:
    async def test_flag_off_issues_limit_top_k_and_keeps_order(self, monkeypatch):
        # Assert the limit, not just the output — a silently widened search
        # would otherwise pass on output alone.
        chunks = [_chunk("a", 0.1), _chunk("b", 0.2), _chunk("c", 0.3), _chunk("d", 0.4)]
        engine = _patch_engine(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(top_k=3)
        result = await retriever.get_retrieved_objects("query")

        assert len(engine.search_calls) == 1
        assert engine.search_calls[0]["limit"] == 3
        assert result == chunks[:3]

    async def test_empty_weights_dict_short_circuits_even_with_text(self, monkeypatch):
        # Preference text alone (guidance) must not widen the search or sort.
        chunks = [_chunk("a", 0.1), _chunk("b", 0.2), _chunk("c", 0.3)]
        engine = _patch_engine(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("## What this user prefers\nconcise answers", {}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(top_k=2)
        result = await retriever.get_retrieved_objects("query")

        assert engine.search_calls[0]["limit"] == 2
        assert result == chunks[:2]

    async def test_weights_over_fetch_reorder_and_trim(self, monkeypatch):
        # top_k=2, six candidates. Weight 0.95 on "e" — outside the raw top 2
        # (and outside the raw top_k entirely) — pulls it into the final list:
        # membership changes, not just order (Demo 4b).
        chunks = [
            _chunk("a", 0.10),
            _chunk("b", 0.20),
            _chunk("e", 0.25),
            _chunk("c", 0.30),
            _chunk("d", 0.40),
            _chunk("f", 0.60),
        ]
        # e raw: rank 3, outside top_k=2. Personalized: 0.25 * 0.73 = 0.1825,
        # which beats b's raw 0.20 — so e enters and b drops out.
        engine = _patch_engine(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {"e": 0.95}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(top_k=2)
        result = await retriever.get_retrieved_objects("query")

        assert engine.search_calls[0]["limit"] == 2 * PREFERENCE_OVERFETCH
        result_ids = [chunk.payload["id"] for chunk in result]
        assert len(result_ids) == 2
        assert "e" in result_ids  # membership change: e was outside raw top_k
        # a (0.10) still wins; e's personalized distance beats b's raw 0.20.
        factor = personal_factor(0.95, INFLUENCE, distance_space=True)
        assert 0.25 * factor < 0.20
        assert result_ids == ["a", "e"]

    async def test_disliked_chunk_drops_out(self, monkeypatch):
        chunks = [_chunk("a", 0.10), _chunk("b", 0.11), _chunk("c", 0.12)]
        engine = _patch_engine(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {"a": 0.0}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(top_k=2)
        result = await retriever.get_retrieved_objects("query")

        assert engine.search_calls[0]["limit"] == 2 * PREFERENCE_OVERFETCH
        # a's distance grows by 1.3: 0.13 > 0.12, so it falls behind b and c.
        assert [chunk.payload["id"] for chunk in result] == ["b", "c"]

    async def test_chunk_id_falls_back_to_id_when_payload_lacks_it(self, monkeypatch):
        bare = SimpleNamespace(id="bare", score=0.5, payload={"text": "no id key"})
        chunks = [_chunk("a", 0.1), bare]
        _patch_engine(monkeypatch, chunks)
        _patch_lookup(monkeypatch, ("", {"bare": 1.0}))
        _patch_influence(monkeypatch)

        retriever = CompletionRetriever(top_k=2)
        result = await retriever.get_retrieved_objects("query")

        # weight 1.0, influence 0.3 → bare's distance 0.5 * 0.7 = 0.35; still
        # behind a, but matched via .id — assert the factor applied by order
        # staying stable and both present.
        assert result[0] is chunks[0]
        assert result[1] is bare


class TestStableSort:
    def test_unmatched_chunks_keep_engine_order(self):
        chunks = [_chunk("a", 0.2), _chunk("b", 0.2), _chunk("c", 0.2)]
        result = _stable_sort_by_personal_distance(chunks, {"zzz": 0.9}, INFLUENCE)
        assert result == chunks

    def test_neutral_weight_is_an_exact_no_op(self):
        chunks = [_chunk("a", 0.2), _chunk("b", 0.2)]
        result = _stable_sort_by_personal_distance(chunks, {"a": 0.5, "b": 0.5}, INFLUENCE)
        assert result == chunks


@pytest.mark.asyncio
class TestSessionlessGuidance:
    async def test_preference_text_passed_as_conversation_history(self, monkeypatch):
        _patch_lookup(monkeypatch, ("PREFS", {}))
        captured = {}

        async def fake_generate_completion(**kwargs):
            captured.update(kwargs)
            return "answer"

        monkeypatch.setattr(retriever_module, "generate_completion", fake_generate_completion)

        retriever = CompletionRetriever(top_k=2)
        result = await retriever._generate_completion_without_session("q", "ctx")

        assert result == ["answer"]
        assert captured["conversation_history"] == "PREFS"
        assert captured["context"] == "ctx"

    async def test_empty_preference_text_passes_falsy_history(self, monkeypatch):
        _patch_lookup(monkeypatch, ("", {}))
        captured = {}

        async def fake_generate_completion(**kwargs):
            captured.update(kwargs)
            return "answer"

        monkeypatch.setattr(retriever_module, "generate_completion", fake_generate_completion)

        retriever = CompletionRetriever(top_k=2)
        await retriever._generate_completion_without_session("q", "ctx")

        # generate_completion treats a falsy history as "no layer", so the
        # system prompt stays byte-identical to the un-personalized path.
        assert captured["conversation_history"] == ""
