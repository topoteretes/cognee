"""Unit tests for the fail-open session embedding helpers and their integration points.

Pure-math helpers (cosine, hybrid QA selection) are tested directly. The ranker's
cosine-vs-overlap relevance and the near-duplicate candidate merge are tested with fakes
and a patched embedder, so no vector engine or LLM is needed.
"""

from datetime import datetime
from uuid import uuid4

import pytest

import cognee.infrastructure.session.session_context_builder as builder_module
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.session_context_builder import (
    DeterministicRanker,
    apply_candidate_updates,
)
from cognee.infrastructure.session.session_context_models import (
    SessionContextEntry,
    normalize_content,
)
from cognee.infrastructure.session.session_embeddings import (
    cosine_similarity,
    select_hybrid_qa_entries,
)


def _qa(question, answer="answer", embedding=None, time_suffix="00"):
    return SessionQAEntry(
        time=f"2026-06-11T10:00:{time_suffix}",
        qa_id=str(uuid4()),
        question=question,
        context="",
        answer=answer,
        embedding=embedding,
    )


def _entry(section, content, **kwargs):
    return SessionContextEntry(
        id=kwargs.pop("id", str(uuid4())),
        section=section,
        content=content,
        normalized_content=normalize_content(content),
        created_at=kwargs.pop("created_at", datetime.utcnow().isoformat()),
        **kwargs,
    )


class FakeSessionManager:
    """In-memory stand-in for SessionManager's context CRUD surface."""

    def __init__(self, entries=None):
        self.store = [
            e.model_dump() if isinstance(e, SessionContextEntry) else e for e in (entries or [])
        ]

    async def get_session_context_entries(self, user_id, session_id):
        return list(self.store)

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        self.store.append(entry_dump)

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_or_mismatched_inputs_return_zero(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestSelectHybridQaEntries:
    def test_no_query_embedding_falls_back_to_last_n(self):
        entries = [_qa(f"q{i}", time_suffix=f"0{i}") for i in range(5)]
        selected = select_hybrid_qa_entries(entries, None, last_n=2)
        assert selected == entries[-2:]

    def test_semantic_recall_prepends_relevant_older_turns(self):
        relevant = _qa("about cats", embedding=[1.0, 0.0], time_suffix="01")
        irrelevant = _qa("about taxes", embedding=[0.0, 1.0], time_suffix="02")
        recent = [_qa(f"recent {i}", time_suffix=f"0{i + 3}") for i in range(2)]
        entries = [relevant, irrelevant] + recent

        selected = select_hybrid_qa_entries(entries, [1.0, 0.0], last_n=2)

        assert selected == [relevant] + recent

    def test_union_is_chronological_and_deduplicated(self):
        older = [
            _qa("a", embedding=[1.0, 0.0], time_suffix="01"),
            _qa("b", embedding=[0.9, 0.1], time_suffix="02"),
        ]
        recent = [_qa("c", embedding=[1.0, 0.0], time_suffix="03")]
        entries = older + recent

        selected = select_hybrid_qa_entries(entries, [1.0, 0.0], last_n=1, semantic_top_k=5)

        # Older relevant turns come first in original order; recent window untouched.
        assert selected == older + recent
        assert len(selected) == len({entry.qa_id for entry in selected})

    def test_similarity_floor_excludes_weak_matches(self):
        weak = _qa("weak", embedding=[0.3, 0.95], time_suffix="01")
        recent = [_qa("recent", time_suffix="02")]
        entries = [weak] + recent

        selected = select_hybrid_qa_entries(entries, [1.0, 0.0], last_n=1, min_similarity=0.5)

        assert selected == recent

    def test_entries_without_embedding_are_never_recalled(self):
        no_embedding = _qa("no embedding", time_suffix="01")
        recent = [_qa("recent", time_suffix="02")]
        selected = select_hybrid_qa_entries([no_embedding] + recent, [1.0, 0.0], last_n=1)
        assert selected == recent

    def test_semantic_top_k_caps_recalled_turns(self):
        older = [_qa(f"q{i}", embedding=[1.0, 0.0], time_suffix=f"0{i}") for i in range(4)]
        recent = [_qa("recent", time_suffix="09")]
        selected = select_hybrid_qa_entries(older + recent, [1.0, 0.0], last_n=1, semantic_top_k=2)
        assert len(selected) == 3  # 2 recalled + 1 recent

    def test_dict_entries_are_supported(self):
        older = _qa("dict entry", embedding=[1.0, 0.0], time_suffix="01").model_dump()
        recent = _qa("recent", time_suffix="02").model_dump()
        selected = select_hybrid_qa_entries([older, recent], [1.0, 0.0], last_n=1)
        assert selected == [older, recent]


class TestRankerRelevance:
    def test_cosine_used_when_both_embeddings_exist(self):
        ranker = DeterministicRanker(query_embedding=[1.0, 0.0])
        aligned = _entry("rules", "completely different words", embedding=[1.0, 0.0])
        misaligned = _entry("rules", "completely different words two", embedding=[0.0, 1.0])

        assert ranker.score(aligned, "query") > ranker.score(misaligned, "query")

    def test_falls_back_to_token_overlap_without_embeddings(self):
        ranker = DeterministicRanker(query_embedding=[1.0, 0.0])
        overlapping = _entry("rules", "graph database tips")
        unrelated = _entry("rules", "something else entirely here")

        assert ranker.score(overlapping, "graph database tips") > ranker.score(
            unrelated, "graph database tips"
        )

    def test_no_query_embedding_matches_legacy_behavior(self):
        legacy = DeterministicRanker()
        entry = _entry("rules", "graph database tips", embedding=[1.0, 0.0])
        assert legacy.score(entry, "graph database tips") == DeterministicRanker(
            query_embedding=None
        ).score(entry, "graph database tips")

    def test_negative_cosine_clamped_to_zero(self):
        ranker = DeterministicRanker(query_embedding=[1.0, 0.0])
        opposite = _entry("rules", "abc", embedding=[-1.0, 0.0])
        neutral = _entry("rules", "abc two", embedding=[0.0, 1.0])
        assert ranker.score(opposite, "zz") == pytest.approx(ranker.score(neutral, "zz"), abs=0.6)


class TestNearDuplicateCandidateMerge:
    @pytest.mark.asyncio
    async def test_near_duplicate_links_instead_of_creating(self, monkeypatch):
        async def fake_embed(text):
            return [1.0, 0.0]

        monkeypatch.setattr(builder_module, "embed_text_safe", fake_embed)

        existing = _entry("rules", "Always answer in formal English.", embedding=[0.99, 0.01])
        manager = FakeSessionManager([existing])

        touched = await apply_candidate_updates(
            session_manager=manager,
            user_id="u",
            session_id="s",
            feedback_entry_id="fb-1",
            candidates=[
                {
                    "section": "rules",
                    "content": "Respond using formal English at all times.",
                    "confidence": 0.9,
                }
            ],
        )

        assert touched == [existing.id]
        assert len(manager.store) == 1
        assert "fb-1" in manager.store[0]["source_feedback_ids"]

    @pytest.mark.asyncio
    async def test_dissimilar_candidate_creates_entry_with_embedding(self, monkeypatch):
        async def fake_embed(text):
            return [0.0, 1.0]

        monkeypatch.setattr(builder_module, "embed_text_safe", fake_embed)

        existing = _entry("rules", "Always answer in formal English.", embedding=[1.0, 0.0])
        manager = FakeSessionManager([existing])

        touched = await apply_candidate_updates(
            session_manager=manager,
            user_id="u",
            session_id="s",
            feedback_entry_id="fb-2",
            candidates=[{"section": "rules", "content": "Cite sources.", "confidence": 0.9}],
        )

        assert len(touched) == 1
        assert touched[0] != existing.id
        assert len(manager.store) == 2
        assert manager.store[1]["embedding"] == [0.0, 1.0]

    @pytest.mark.asyncio
    async def test_embedding_failure_degrades_to_exact_match_only(self, monkeypatch):
        async def fake_embed(text):
            return None

        monkeypatch.setattr(builder_module, "embed_text_safe", fake_embed)

        existing = _entry("rules", "Always answer in formal English.", embedding=[1.0, 0.0])
        manager = FakeSessionManager([existing])

        touched = await apply_candidate_updates(
            session_manager=manager,
            user_id="u",
            session_id="s",
            feedback_entry_id="fb-3",
            candidates=[
                {
                    "section": "rules",
                    "content": "always answer in formal english.",
                    "confidence": 0.9,
                },
                {"section": "rules", "content": "Reworded formal English rule.", "confidence": 0.9},
            ],
        )

        # Exact (normalized) dup links; the reworded one cannot be matched without embeddings.
        assert touched[0] == existing.id
        assert len(manager.store) == 2
