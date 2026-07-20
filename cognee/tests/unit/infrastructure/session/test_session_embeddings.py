"""Unit tests for session QA vector recall and active-context helpers."""

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.session.session_context_builder import (
    DeterministicRanker,
    apply_candidate_updates,
)
from cognee.infrastructure.session.session_context_models import (
    SessionContextEntry,
    normalize_content,
)
from cognee.infrastructure.session.session_embeddings import (
    SESSION_QA_VECTOR_COLLECTION,
    delete_session_qa_vector,
    delete_session_qa_vectors,
    index_session_qa,
    merge_hybrid_qa_entries,
    search_session_qa_ids,
    select_hybrid_qa_entries,
    session_scope_tag,
)
from cognee.infrastructure.session.session_turn import select_session_history


def _qa(question, answer="answer", qa_id=None, time_suffix="00"):
    return SessionQAEntry(
        time=f"2026-06-11T10:00:{time_suffix}",
        qa_id=qa_id or str(uuid4()),
        question=question,
        context="",
        answer=answer,
    )


def _entry(section, content, **kwargs):
    return SessionContextEntry(
        id=kwargs.pop("id", str(uuid4())),
        section=section,
        content=content,
        normalized_content=normalize_content(content),
        created_at=kwargs.pop("created_at", datetime.now(timezone.utc).isoformat()),
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


class FakeHistoryManager:
    """In-memory stand-in for SessionManager's QA history surface."""

    session_history_last_n = 1

    def __init__(self, entries):
        self.entries = list(entries)

    async def get_session(
        self,
        *,
        user_id,
        session_id,
        formatted=False,
        last_n=None,
        include_context=True,
    ):
        entries = self.entries[-last_n:] if last_n is not None else self.entries
        if formatted:
            return self.format_entries(entries, include_context=include_context)
        return list(entries)

    async def get_session_entries_by_ids(self, *, user_id, session_id, qa_ids):
        wanted_ids = set(qa_ids)
        return [entry for entry in self.entries if entry["qa_id"] in wanted_ids]

    def format_entries(self, entries, include_context=False):
        return "\n".join(
            f"User: {entry['question']}\nAssistant: {entry['answer']}" for entry in entries
        )


def test_session_qa_entry_has_no_embedding_field():
    entry = _qa("question")

    assert not hasattr(entry, "embedding")
    assert "embedding" not in entry.model_dump()


class TestSelectHybridQaEntries:
    def test_no_vector_ids_falls_back_to_last_n(self):
        entries = [_qa(f"q{i}", time_suffix=f"0{i}") for i in range(5)]
        selected = select_hybrid_qa_entries(entries, None, last_n=2)
        assert selected == entries[-2:]

    def test_vector_ids_recall_older_turns(self):
        relevant = _qa("about cats", qa_id="qa-relevant", time_suffix="01")
        irrelevant = _qa("about taxes", qa_id="qa-irrelevant", time_suffix="02")
        recent = [_qa(f"recent {i}", time_suffix=f"0{i + 3}") for i in range(2)]
        entries = [relevant, irrelevant] + recent

        selected = select_hybrid_qa_entries(entries, ["qa-relevant"], last_n=2)

        assert selected == [relevant] + recent

    def test_union_is_chronological_and_deduplicated(self):
        older = [
            _qa("a", qa_id="qa-a", time_suffix="01"),
            _qa("b", qa_id="qa-b", time_suffix="02"),
        ]
        recent = [_qa("c", qa_id="qa-c", time_suffix="03")]
        entries = older + recent

        selected = select_hybrid_qa_entries(entries, ["qa-c", "qa-b", "qa-a"], last_n=1)

        # Selected turns come back in original order; recent window is not duplicated.
        assert selected == older + recent
        assert len(selected) == len({entry.qa_id for entry in selected})

    def test_dict_entries_are_supported(self):
        older = _qa("dict entry", qa_id="qa-dict", time_suffix="01").model_dump()
        recent = _qa("recent", time_suffix="02").model_dump()
        selected = select_hybrid_qa_entries([older, recent], ["qa-dict"], last_n=1)
        assert selected == [older, recent]

    def test_merge_hybrid_entries_deduplicates_and_orders_chronologically(self):
        older = _qa("old", qa_id="qa-old", time_suffix="01").model_dump()
        middle = _qa("middle", qa_id="qa-middle", time_suffix="02").model_dump()
        recent = _qa("recent", qa_id="qa-recent", time_suffix="03").model_dump()

        selected = merge_hybrid_qa_entries(
            recent_entries=[middle, recent],
            vector_entries=[older, recent],
        )

        assert selected == [older, middle, recent]

    @pytest.mark.asyncio
    async def test_session_history_hydrates_vector_hits_by_id(self, monkeypatch):
        entries = [
            _qa("current session old", qa_id="current-old", time_suffix="01").model_dump(),
            _qa("current session middle", qa_id="current-middle", time_suffix="02").model_dump(),
            _qa("current session recent", qa_id="current-recent", time_suffix="03").model_dump(),
        ]
        manager = FakeHistoryManager(entries)
        search = AsyncMock(return_value=["current-old"])
        monkeypatch.setattr(
            "cognee.infrastructure.session.session_turn.search_session_qa_ids", search
        )

        history = await select_session_history(
            manager,
            user_id="u1",
            session_id="s1",
            query_text="old",
        )

        assert "current session old" in history
        assert "current session recent" in history
        assert "current session middle" not in history


class TestSessionQaVectorHelpers:
    @pytest.mark.asyncio
    async def test_index_session_qa_uses_session_scope_tag(self, monkeypatch):
        indexed = []

        async def fake_index_data_points(points):
            indexed.extend(points)

        index_module = importlib.import_module("cognee.tasks.storage.index_data_points")
        monkeypatch.setattr(index_module, "index_data_points", fake_index_data_points)
        qa_id = str(uuid4())

        await index_session_qa(
            user_id="u1",
            session_id="s1",
            qa_id=qa_id,
            question="Question?",
            answer="Answer.",
        )

        assert len(indexed) == 1
        assert str(indexed[0].id) == qa_id
        assert indexed[0].text == "Question?\nAnswer."
        assert indexed[0].belongs_to_set == [session_scope_tag("u1", "s1")]

    @pytest.mark.asyncio
    async def test_search_session_qa_ids_uses_vector_engine_scope(self, monkeypatch):
        result_id = uuid4()
        vector_engine = SimpleNamespace()

        async def fake_search(*args, **kwargs):
            vector_engine.search_call = (args, kwargs)
            return [SimpleNamespace(id=result_id)]

        vector_engine.search = fake_search

        async def _get_vector_engine():
            return vector_engine

        monkeypatch.setattr(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _get_vector_engine,
        )

        qa_ids = await search_session_qa_ids(
            user_id="u1",
            session_id="s1",
            query_text="Question?",
            limit=3,
        )

        assert qa_ids == [str(result_id)]
        args, kwargs = vector_engine.search_call
        assert args == (SESSION_QA_VECTOR_COLLECTION,)
        assert kwargs["query_text"] == "Question?"
        assert kwargs["query_vector"] is None
        assert kwargs["limit"] == 3
        assert kwargs["node_name"] == [session_scope_tag("u1", "s1")]

    @pytest.mark.asyncio
    async def test_search_session_qa_ids_treats_missing_collection_as_empty(self, monkeypatch):
        vector_engine = SimpleNamespace()

        async def fake_search(*_args, **_kwargs):
            raise CollectionNotFoundError("Collection not found")

        vector_engine.search = fake_search

        async def _get_vector_engine():
            return vector_engine

        monkeypatch.setattr(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _get_vector_engine,
        )

        qa_ids = await search_session_qa_ids(
            user_id="u1",
            session_id="s1",
            query_text="Question?",
            limit=3,
        )

        assert qa_ids == []

    @pytest.mark.asyncio
    async def test_delete_session_qa_vector_uses_qa_id(self, monkeypatch):
        qa_id = uuid4()
        vector_engine = SimpleNamespace()

        async def fake_delete_data_points(collection_name, data_point_ids):
            vector_engine.delete_call = (collection_name, data_point_ids)

        vector_engine.delete_data_points = fake_delete_data_points

        async def _get_vector_engine():
            return vector_engine

        monkeypatch.setattr(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _get_vector_engine,
        )

        await delete_session_qa_vector(qa_id=str(qa_id))

        collection_name, data_point_ids = vector_engine.delete_call
        assert collection_name == SESSION_QA_VECTOR_COLLECTION
        assert data_point_ids == [qa_id]

    @pytest.mark.asyncio
    async def test_delete_session_qa_vectors_removes_session_scope_tag(self, monkeypatch):
        vector_engine = SimpleNamespace()

        async def fake_remove_belongs_to_set_tags(tags):
            vector_engine.remove_tags_call = tags

        vector_engine.remove_belongs_to_set_tags = fake_remove_belongs_to_set_tags

        async def _get_vector_engine():
            return vector_engine

        monkeypatch.setattr(
            "cognee.infrastructure.databases.vector.get_vector_engine_async",
            _get_vector_engine,
        )

        await delete_session_qa_vectors(user_id="u1", session_id="s1")

        assert vector_engine.remove_tags_call == [session_scope_tag("u1", "s1")]


class TestRankerRelevance:
    def test_token_overlap_boosts_relevant_entries(self):
        ranker = DeterministicRanker()
        overlapping = _entry("rules", "graph database tips")
        unrelated = _entry("rules", "something else entirely here")

        assert ranker.score(overlapping, "graph database tips") > ranker.score(
            unrelated, "graph database tips"
        )

    def test_stored_embeddings_do_not_affect_ranking(self):
        ranker = DeterministicRanker()
        aligned = _entry("rules", "same words", embedding=[1.0, 0.0])
        misaligned = _entry("rules", "same words", embedding=[0.0, 1.0])

        assert ranker.score(aligned, "same words") == ranker.score(misaligned, "same words")


class TestCandidateExactDuplicateMerge:
    @pytest.mark.asyncio
    async def test_exact_duplicate_links_instead_of_creating(self):
        existing = _entry("rules", "Always answer in formal English.")
        manager = FakeSessionManager([existing])

        touched = await apply_candidate_updates(
            session_manager=manager,
            user_id="u",
            session_id="s",
            source_id="fb-1",
            candidates=[
                {
                    "section": "rules",
                    "content": "always answer in formal english.",
                    "confidence": 0.9,
                }
            ],
        )

        assert touched == [existing.id]
        assert len(manager.store) == 1
        assert "fb-1" in manager.store[0]["source_feedback_ids"]

    @pytest.mark.asyncio
    async def test_reworded_candidate_creates_entry_without_embedding(self):
        existing = _entry("rules", "Always answer in formal English.")
        manager = FakeSessionManager([existing])

        touched = await apply_candidate_updates(
            session_manager=manager,
            user_id="u",
            session_id="s",
            source_id="fb-2",
            candidates=[{"section": "rules", "content": "Cite sources.", "confidence": 0.9}],
        )

        assert len(touched) == 1
        assert touched[0] != existing.id
        assert len(manager.store) == 2
        assert manager.store[1].get("embedding") is None

    @pytest.mark.asyncio
    async def test_exact_and_reworded_candidates_are_handled_separately(self):
        existing = _entry("rules", "Always answer in formal English.")
        manager = FakeSessionManager([existing])

        touched = await apply_candidate_updates(
            session_manager=manager,
            user_id="u",
            session_id="s",
            source_id="fb-3",
            candidates=[
                {
                    "section": "rules",
                    "content": "always answer in formal english.",
                    "confidence": 0.9,
                },
                {"section": "rules", "content": "Reworded formal English rule.", "confidence": 0.9},
            ],
        )

        assert touched[0] == existing.id
        assert len(manager.store) == 2
