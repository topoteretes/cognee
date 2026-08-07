"""Contract for where concurrent mode runs and what one turn costs.

Three guarantees are pinned here:

* dispatch happens only at the complete-operation boundaries (``get_retriever_output``
  and ``get_completion``); partial retriever methods keep their current behavior,
* every unsupported input falls back to the sequential path before anything is written,
* a turn costs one answer call, with the turn analysis running alongside it.
"""

import asyncio
import inspect
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_search_models import SessionTurnSnapshot
from cognee.modules.retrieval import session_search
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.session_search import try_concurrent_turn
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType

SUPPORTED_RETRIEVERS = [
    CompletionRetriever,
    GraphCompletionRetriever,
    GraphSummaryCompletionRetriever,
    HybridRetriever,
    TripletRetriever,
]

# Partial operations: they cannot satisfy the dual-retrieval contract, so they must never
# reach the orchestrator no matter which mode is configured.
PARTIAL_METHODS = [
    "get_retrieved_objects",
    "get_context_from_objects",
    "get_completion_from_context",
]


class FakeSessionManager:
    """Session facade recording exactly what a concurrent turn persists."""

    def __init__(self, *, available=True, auto_feedback=True):
        self.available = available
        self.auto_feedback = auto_feedback
        self.dataset_id = None
        self.qas = []
        self.context_entries = []
        self.updates = []

    def is_session_available_for_completion(self, user_id):
        return self.available

    def is_auto_feedback_enabled(self):
        return self.auto_feedback

    def resolve_session_id(self, session_id):
        return session_id or "resolved-session"

    async def add_qa(self, **kwargs):
        self.qas.append(kwargs)
        return f"qa-{len(self.qas)}"

    async def create_session_context_entry(self, *, entry_dump, **kwargs):
        self.entries.append(entry_dump)
        return True

    async def get_session_context_entries(self, strict=False, **kwargs):
        return list(self.entries)

    async def update_session_context_entry(self, *, entry_id, merge, **kwargs):
        self.updates.append((entry_id, merge))
        for entry in self.context_entries:
            if entry.get("id") == entry_id:
                entry.update(merge)
                return True
        return False


@pytest.fixture
def concurrent_env(monkeypatch):
    """Concurrent mode with every collaborator faked except the completion path itself."""
    state = SimpleNamespace(
        manager=FakeSessionManager(),
        mode=session_search.CONCURRENT_MODE,
        user=SimpleNamespace(id=uuid4()),
        llm_calls=[],
    )

    async def fake_llm(text_input, system_prompt, response_model, **kwargs):
        state.llm_calls.append(response_model)
        if response_model is SessionTurnAnalysis:
            return SessionTurnAnalysis()
        if response_model is str:
            return "answer"
        return response_model(text="answer")

    monkeypatch.setattr(
        session_search,
        "CacheConfig",
        lambda: SimpleNamespace(session_search_mode=state.mode),
    )
    monkeypatch.setattr(session_search, "session_user", SimpleNamespace(get=lambda: state.user))
    monkeypatch.setattr(session_search, "get_session_manager", lambda: state.manager)
    monkeypatch.setattr(
        session_search,
        "load_turn_snapshot",
        AsyncMock(return_value=SessionTurnSnapshot(raw_message="question")),
    )
    monkeypatch.setattr(
        session_search, "update_node_access_timestamps", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", staticmethod(fake_llm))
    return state


def build_retriever(retriever_class, **kwargs):
    """One supported retriever with only its database calls stubbed out."""
    retriever = retriever_class(session_id="s1", **kwargs)
    results = [{"id": "n1", "text": "n1"}]
    if isinstance(retriever, HybridRetriever):
        results = {"chunks": results, "entities": [], "facts": []}
    retriever.get_retrieved_objects = AsyncMock(return_value=results)
    retriever.get_context_from_objects = AsyncMock(return_value="context")
    retriever.append_references = AsyncMock(side_effect=lambda answers, objects: answers)
    return retriever


class TestModeBoundary:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("retriever_class", SUPPORTED_RETRIEVERS)
    async def test_get_completion_answers_from_the_concurrent_turn(
        self, retriever_class, concurrent_env
    ):
        completion = await build_retriever(retriever_class).get_completion(query="question")

        assert completion == ["answer"]
        assert len(concurrent_env.manager.qas) == 1

    @pytest.mark.asyncio
    async def test_sequential_mode_never_enters_the_concurrent_turn(self, concurrent_env):
        concurrent_env.mode = "sequential"

        assert (
            await try_concurrent_turn(build_retriever(CompletionRetriever), raw_query="question")
            is None
        )
        assert concurrent_env.llm_calls == []
        assert concurrent_env.manager.qas == []

    @pytest.mark.asyncio
    async def test_get_retriever_output_maps_the_turn_result(self, concurrent_env, monkeypatch):
        # The package re-exports the function under the module's own name.
        module = import_module("cognee.modules.search.methods.get_retriever_output")

        retriever = build_retriever(CompletionRetriever)
        monkeypatch.setattr(
            module,
            "get_graph_engine",
            AsyncMock(return_value=SimpleNamespace(is_empty=AsyncMock(return_value=False))),
        )
        monkeypatch.setattr(
            module, "get_search_type_retriever_instance", AsyncMock(return_value=retriever)
        )

        payload = await module.get_retriever_output(SearchType.RAG_COMPLETION, "question")

        assert payload.completion == ["answer"]
        assert payload.context == "context"
        retriever.get_context_from_objects.assert_awaited_once()

    def test_partial_operations_do_not_dispatch(self):
        for retriever_class in [BaseRetriever, *SUPPORTED_RETRIEVERS]:
            for name in PARTIAL_METHODS:
                method = getattr(retriever_class, name, None)
                if method is None:
                    continue
                assert "try_concurrent_turn" not in inspect.getsource(method), (
                    f"{retriever_class.__name__}.{name} is a partial operation"
                )


class TestUnsupportedInputsFallBack:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call_kwargs",
        [
            pytest.param({"is_batch": True}, id="batch"),
            pytest.param({"only_context": True}, id="only_context"),
            pytest.param({"original_search_type": SearchType.FEELING_LUCKY}, id="feeling_lucky"),
        ],
    )
    async def test_unsupported_call_shapes(self, call_kwargs, concurrent_env):
        result = await try_concurrent_turn(
            build_retriever(CompletionRetriever), raw_query="question", **call_kwargs
        )

        assert result is None
        assert concurrent_env.manager.qas == []

    @pytest.mark.asyncio
    async def test_retriever_subclasses_are_not_supported(self, concurrent_env):
        class CustomGraphRetriever(GraphCompletionRetriever):
            pass

        assert (
            await try_concurrent_turn(build_retriever(CustomGraphRetriever), raw_query="question")
            is None
        )

    @pytest.mark.asyncio
    async def test_missing_user_or_unavailable_session(self, concurrent_env):
        retriever = build_retriever(CompletionRetriever)

        concurrent_env.user = None
        assert await try_concurrent_turn(retriever, raw_query="question") is None

        concurrent_env.user = SimpleNamespace(id=uuid4())
        concurrent_env.manager.available = False
        assert await try_concurrent_turn(retriever, raw_query="question") is None

        assert concurrent_env.llm_calls == []


class TestTurnCost:
    @pytest.mark.asyncio
    async def test_turn_costs_one_answer_call_plus_the_analysis(self, concurrent_env):
        result = await try_concurrent_turn(
            build_retriever(CompletionRetriever), raw_query="question"
        )

        assert result.completion == ["answer"]
        # Exactly two calls: the caller's own answer model, and the turn analysis.
        # Order is not asserted — the lanes are concurrent, so either may land first.
        assert len(concurrent_env.llm_calls) == 2
        assert set(concurrent_env.llm_calls) == {str, SessionTurnAnalysis}
        assert len(concurrent_env.manager.qas) == 1

    @pytest.mark.asyncio
    async def test_auto_feedback_off_answers_without_analyzing(self, concurrent_env):
        concurrent_env.manager.auto_feedback = False

        result = await try_concurrent_turn(
            build_retriever(CompletionRetriever), raw_query="question"
        )

        assert result.completion == ["answer"]
        assert concurrent_env.llm_calls == [str]
        assert concurrent_env.manager.context_entries == []

    @pytest.mark.asyncio
    async def test_rapid_turns_in_one_session_are_serialized(self, concurrent_env):
        retriever = build_retriever(CompletionRetriever)
        peak = 0
        active = 0
        original = session_search.retrieve_turn_context

        async def counted(*args, **kwargs):
            nonlocal peak, active
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            try:
                return await original(*args, **kwargs)
            finally:
                active -= 1

        session_search.retrieve_turn_context = counted
        try:
            await asyncio.gather(
                try_concurrent_turn(retriever, raw_query="first"),
                try_concurrent_turn(retriever, raw_query="second"),
            )
        finally:
            session_search.retrieve_turn_context = original

        assert peak == 1
        assert len(concurrent_env.manager.qas) == 2


def test_session_facade_stays_free_of_search_mode_state():
    """Ownership boundaries: no search-mode state on the facade, no wrapper leakage."""
    from cognee.infrastructure.session import session_manager
    from cognee.modules.retrieval.utils import references

    session_manager_source = inspect.getsource(session_manager)
    assert "session_search_mode" not in session_manager_source
    assert "try_concurrent_turn" not in session_manager_source

    # Generic reference helpers keep receiving plain strings and lists.
    assert "session_search_models" not in inspect.getsource(references)
