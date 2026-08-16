"""Unit tests for Phase 5 of user preferences: the graph completion retriever.

Covers the deterministic wiring only — mocked lookup, mocked triplet search,
mocked completion utils, no graph database, no LLM:
- ``get_triplets`` forwards the memoized weight map into
  ``brute_force_triplet_search`` as ``personal_weights``, and an empty map is
  forwarded as ``None`` — byte-identical to an un-personalized run.
- The sessionless completion passes the preference text as
  ``conversation_history`` to BOTH arms: ``generate_completion`` for a single
  query and ``generate_completion_batch`` for a query batch.

The scoring math itself (``_personal_distance`` eligibility and the
multiplicative compose with the feedback blend) is covered in
``tests/unit/modules/graph/cognee_graph_test.py``.
"""

from typing import Dict, Tuple

import pytest

import cognee.modules.retrieval.graph_completion_retriever as retriever_module
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


def _patch_lookup(monkeypatch, result: Tuple[str, Dict[str, float]]):
    async def fake_load_active_preferences():
        return result

    monkeypatch.setattr(retriever_module, "load_active_preferences", fake_load_active_preferences)


@pytest.mark.asyncio
class TestGetTriplets:
    async def test_weights_forwarded_to_brute_force_search(self, monkeypatch):
        _patch_lookup(monkeypatch, ("", {"node-1": 0.9}))
        captured = {}

        async def fake_search(*args, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(retriever_module, "brute_force_triplet_search", fake_search)

        retriever = GraphCompletionRetriever(top_k=3)
        result = await retriever.get_triplets(query="q")

        assert result == []
        assert captured["personal_weights"] == {"node-1": 0.9}
        assert captured["top_k"] == 3

    async def test_empty_weights_forwarded_as_none(self, monkeypatch):
        # Flag off / no node / lookup failure all yield {} from the lookup;
        # the search must then see personal_weights=None, the pre-change shape.
        _patch_lookup(monkeypatch, ("", {}))
        captured = {}

        async def fake_search(*args, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(retriever_module, "brute_force_triplet_search", fake_search)

        retriever = GraphCompletionRetriever(top_k=3)
        await retriever.get_triplets(query="q")

        assert captured["personal_weights"] is None


@pytest.mark.asyncio
class TestSessionlessGuidance:
    async def test_preference_text_reaches_single_arm(self, monkeypatch):
        _patch_lookup(monkeypatch, ("PREFS", {}))
        captured = {}

        async def fake_generate_completion(**kwargs):
            captured.update(kwargs)
            return "answer"

        monkeypatch.setattr(retriever_module, "generate_completion", fake_generate_completion)

        retriever = GraphCompletionRetriever(top_k=3)
        result = await retriever._generate_completion_without_session("q", None, "ctx")

        assert result == ["answer"]
        assert captured["conversation_history"] == "PREFS"
        assert captured["context"] == "ctx"

    async def test_preference_text_reaches_batch_arm(self, monkeypatch):
        _patch_lookup(monkeypatch, ("PREFS", {}))
        captured = {}

        async def fake_generate_completion_batch(**kwargs):
            captured.update(kwargs)
            return ["a1", "a2"]

        monkeypatch.setattr(
            retriever_module, "generate_completion_batch", fake_generate_completion_batch
        )

        retriever = GraphCompletionRetriever(top_k=3)
        result = await retriever._generate_completion_without_session(None, ["q1", "q2"], "ctx")

        assert result == ["a1", "a2"]
        assert captured["conversation_history"] == "PREFS"
        assert captured["query_batch"] == ["q1", "q2"]

    async def test_empty_preference_text_passes_falsy_history(self, monkeypatch):
        _patch_lookup(monkeypatch, ("", {}))
        captured = {}

        async def fake_generate_completion(**kwargs):
            captured.update(kwargs)
            return "answer"

        monkeypatch.setattr(retriever_module, "generate_completion", fake_generate_completion)

        retriever = GraphCompletionRetriever(top_k=3)
        await retriever._generate_completion_without_session("q", None, "ctx")

        # generate_completion treats a falsy history as "no layer", so the
        # system prompt stays byte-identical to the un-personalized path.
        assert captured["conversation_history"] == ""
