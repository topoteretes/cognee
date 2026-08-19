"""Contract for where concurrent mode runs and what one turn costs.

Three guarantees are pinned here:

* ``get_retriever_output`` calls only ``run_session_aware_completion``; that door
  dispatches to concurrent or sequential,
* retriever ``get_completion`` stays as on ``dev``,
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
from cognee.infrastructure.session.session_concurrent_turn import SessionTurnContext
from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.retrieval import session_aware_completion
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.session_aware_completion import (
    run_concurrent_session_turn,
    run_session_aware_completion,
    should_run_concurrent,
)
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType

SUPPORTED_RETRIEVERS = [
    CompletionRetriever,
    GraphCompletionRetriever,
    HybridRetriever,
    TripletRetriever,
]

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
        self.entries = self.context_entries

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
        mode=session_aware_completion.CONCURRENT_MODE,
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
        session_aware_completion,
        "CacheConfig",
        lambda: SimpleNamespace(session_search_mode=state.mode),
    )
    monkeypatch.setattr(
        session_aware_completion, "session_user", SimpleNamespace(get=lambda: state.user)
    )
    monkeypatch.setattr(session_aware_completion, "get_session_manager", lambda: state.manager)
    monkeypatch.setattr(
        session_aware_completion,
        "load_turn_context",
        AsyncMock(return_value=SessionTurnContext(raw_message="question")),
    )
    monkeypatch.setattr(
        session_aware_completion, "update_node_access_timestamps", AsyncMock(return_value=None)
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
    # run_sequential_session_turn calls inspect.signature() on this to check for
    # effective_query/turn_preparation kwargs. On Python 3.10, an un-spec'd Mock crashes
    # there, and spec=<bound method> crashes too (inspect follows its mocked __func__).
    # Setting __signature__ directly is the one path inspect.signature() short-circuits on.
    completion_mock = AsyncMock(return_value=["answer"])
    completion_mock.__signature__ = inspect.signature(retriever.get_completion_from_context)
    retriever.get_completion_from_context = completion_mock
    retriever.prepare_session_turn_for_retrieval = AsyncMock(
        return_value=SessionTurnPreparation(should_answer=True, effective_query="question")
    )
    retriever.append_references = AsyncMock(side_effect=lambda answers, objects: answers)
    return retriever


class TestModeBoundary:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("retriever_class", SUPPORTED_RETRIEVERS)
    async def test_get_completion_stays_on_the_old_flow(self, retriever_class, concurrent_env):
        completion = await build_retriever(retriever_class).get_completion(query="question")

        assert completion == ["answer"]
        assert concurrent_env.manager.qas == []
        assert SessionTurnAnalysis not in concurrent_env.llm_calls

    @pytest.mark.asyncio
    async def test_sequential_mode_uses_the_sequential_runner(self, concurrent_env, monkeypatch):
        concurrent_env.mode = "sequential"
        retriever = build_retriever(CompletionRetriever)
        concurrent_runner = AsyncMock(side_effect=AssertionError("concurrent runner must not run"))
        monkeypatch.setattr(
            session_aware_completion, "run_concurrent_session_turn", concurrent_runner
        )

        assert should_run_concurrent(retriever, original_search_type=None) is False
        objects, context, completion = await run_session_aware_completion(
            retriever, raw_query="question"
        )

        concurrent_runner.assert_not_awaited()
        assert objects == [{"id": "n1", "text": "n1"}] or objects == {
            "chunks": [{"id": "n1", "text": "n1"}],
            "entities": [],
            "facts": [],
        }
        assert context == "context"
        assert completion == ["answer"]
        assert concurrent_env.manager.qas == []

    @pytest.mark.asyncio
    async def test_get_retriever_output_uses_only_the_door(self, concurrent_env, monkeypatch):
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
        assert len(concurrent_env.manager.qas) == 1

    def test_get_completion_does_not_call_the_door(self):
        for retriever_class in [BaseRetriever, GraphCompletionRetriever, HybridRetriever]:
            source = inspect.getsource(retriever_class.get_completion)
            assert "run_session_aware_completion" not in source

    def test_partial_operations_do_not_dispatch(self):
        for retriever_class in [BaseRetriever, *SUPPORTED_RETRIEVERS]:
            for name in PARTIAL_METHODS:
                method = getattr(retriever_class, name, None)
                if method is None:
                    continue
                source = inspect.getsource(method)
                assert "run_session_aware_completion" not in source

    def test_search_output_calls_only_the_public_door(self):
        output_source = inspect.getsource(
            import_module("cognee.modules.search.methods.get_retriever_output")
        )
        assert "run_session_aware_completion" in output_source
        assert "run_concurrent_session_turn" not in output_source
        assert "run_sequential_session_turn" not in output_source
        assert "try_concurrent" not in output_source


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
    async def test_unsupported_call_shapes(self, call_kwargs, concurrent_env, monkeypatch):
        retriever = build_retriever(CompletionRetriever)
        concurrent_runner = AsyncMock(side_effect=AssertionError("concurrent runner must not run"))
        monkeypatch.setattr(
            session_aware_completion, "run_concurrent_session_turn", concurrent_runner
        )

        assert should_run_concurrent(retriever, **call_kwargs) is False
        objects, context, completion = await run_session_aware_completion(
            retriever, raw_query="question", **call_kwargs
        )

        concurrent_runner.assert_not_awaited()
        if call_kwargs.get("only_context"):
            assert completion is None
            assert context == "context"
        else:
            assert completion == ["answer"]
        assert concurrent_env.manager.qas == []

    @pytest.mark.asyncio
    async def test_retriever_subclasses_are_not_supported(self, concurrent_env, monkeypatch):
        class CustomGraphRetriever(GraphCompletionRetriever):
            pass

        retriever = build_retriever(CustomGraphRetriever)
        concurrent_runner = AsyncMock(side_effect=AssertionError("concurrent runner must not run"))
        monkeypatch.setattr(
            session_aware_completion, "run_concurrent_session_turn", concurrent_runner
        )

        assert should_run_concurrent(retriever) is False
        _objects, _context, completion = await run_session_aware_completion(
            retriever, raw_query="question"
        )
        concurrent_runner.assert_not_awaited()
        assert completion == ["answer"]
        assert concurrent_env.manager.qas == []

    @pytest.mark.asyncio
    async def test_missing_user_or_unavailable_session(self, concurrent_env):
        retriever = build_retriever(CompletionRetriever)

        concurrent_env.user = None
        assert should_run_concurrent(retriever) is False

        concurrent_env.user = SimpleNamespace(id=uuid4())
        concurrent_env.manager.available = False
        assert should_run_concurrent(retriever) is False

        assert concurrent_env.llm_calls == []


class TestTurnCost:
    @pytest.mark.asyncio
    async def test_turn_costs_one_answer_call_plus_the_analysis(self, concurrent_env):
        _objects, context, completion = await run_session_aware_completion(
            build_retriever(CompletionRetriever), raw_query="question"
        )

        assert completion == ["answer"]
        assert context == "context"
        assert len(concurrent_env.llm_calls) == 2
        assert set(concurrent_env.llm_calls) == {str, SessionTurnAnalysis}
        assert len(concurrent_env.manager.qas) == 1

    @pytest.mark.asyncio
    async def test_auto_feedback_off_answers_without_analyzing(self, concurrent_env):
        concurrent_env.manager.auto_feedback = False

        _objects, _context, completion = await run_session_aware_completion(
            build_retriever(CompletionRetriever), raw_query="question"
        )

        assert completion == ["answer"]
        assert concurrent_env.llm_calls == [str]
        assert concurrent_env.manager.context_entries == []

    @pytest.mark.asyncio
    async def test_rapid_turns_in_one_session_are_serialized(self, concurrent_env):
        retriever = build_retriever(CompletionRetriever)
        peak = 0
        active = 0
        original = session_aware_completion._retrieve_merged_objects

        async def counted(*args, **kwargs):
            nonlocal peak, active
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            try:
                return await original(*args, **kwargs)
            finally:
                active -= 1

        session_aware_completion._retrieve_merged_objects = counted
        try:
            await asyncio.gather(
                run_concurrent_session_turn(retriever, raw_query="first"),
                run_concurrent_session_turn(retriever, raw_query="second"),
            )
        finally:
            session_aware_completion._retrieve_merged_objects = original

        assert peak == 1
        assert len(concurrent_env.manager.qas) == 2


class TestNoAnswerTurns:
    """SDK-402: every session turn, answered or not, lands in QA history."""

    @pytest.mark.asyncio
    async def test_sequential_no_answer_turn_records_acknowledgement_qa(
        self, concurrent_env, monkeypatch
    ):
        concurrent_env.mode = "sequential"
        retriever = build_retriever(CompletionRetriever)
        retriever.prepare_session_turn_for_retrieval = AsyncMock(
            return_value=SessionTurnPreparation(should_answer=False, response_to_user="Got it.")
        )

        objects, context, completion = await run_session_aware_completion(
            retriever, raw_query="thanks, that was helpful!"
        )

        assert objects is None
        assert context is None
        assert completion == ["Got it."]
        retriever.get_retrieved_objects.assert_not_awaited()
        assert len(concurrent_env.manager.qas) == 1
        qa = concurrent_env.manager.qas[0]
        assert qa["question"] == "thanks, that was helpful!"
        assert qa["answer"] == "Got it."

    @pytest.mark.asyncio
    async def test_concurrent_no_answer_turn_stores_and_returns_acknowledgement(
        self, concurrent_env, monkeypatch
    ):
        """The answer lane's generated text must not be returned or stored once the
        analysis lane decides the turn is feedback-only."""
        monkeypatch.setattr(
            session_aware_completion,
            "load_turn_context",
            AsyncMock(
                return_value=SessionTurnContext(
                    raw_message="thanks, that was helpful!",
                    previous_qa_id="qa-1",
                    previous_question="What is X?",
                    previous_answer="X is Y.",
                )
            ),
        )

        async def fake_llm(text_input, system_prompt, response_model, **kwargs):
            concurrent_env.llm_calls.append(response_model)
            if response_model is SessionTurnAnalysis:
                return SessionTurnAnalysis(response_to_user="Got it.")
            if response_model is str:
                return "a generated answer that must be discarded"
            return response_model(text="a generated answer that must be discarded")

        monkeypatch.setattr(LLMGateway, "acreate_structured_output", staticmethod(fake_llm))
        retriever = build_retriever(CompletionRetriever)

        _objects, _context, completion = await run_session_aware_completion(
            retriever, raw_query="thanks, that was helpful!"
        )

        assert completion == ["Got it."]
        retriever.append_references.assert_not_awaited()
        assert len(concurrent_env.manager.qas) == 1
        qa = concurrent_env.manager.qas[0]
        assert qa["question"] == "thanks, that was helpful!"
        assert qa["answer"] == "Got it."


def test_session_facade_stays_free_of_search_mode_state():
    """Ownership boundaries: no search-mode state on the facade, no wrapper leakage."""
    from cognee.infrastructure.session import session_manager
    from cognee.modules.retrieval.utils import references

    session_manager_source = inspect.getsource(session_manager)
    assert "session_search_mode" not in session_manager_source
    assert "run_session_aware_completion" not in session_manager_source

    assert "SessionTurnContext" not in inspect.getsource(references)
