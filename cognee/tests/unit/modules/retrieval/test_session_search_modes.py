"""Contract for where latency mode runs and what one turn costs.

Three guarantees are pinned here:

* dispatch happens only at the complete-operation boundaries (``get_retriever_output``
  and ``get_completion``); partial retriever methods keep their current behavior,
* every unsupported input falls back to the accuracy path before anything is written,
* one latency turn blocks on exactly one LLM completion — maintenance is never awaited.
"""

import asyncio
import inspect
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.session import session_maintenance, session_maintenance_worker
from cognee.infrastructure.session.session_search_models import (
    SessionMaintenanceResult,
    SessionTurnSnapshot,
)
from cognee.modules.retrieval import session_search
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.session_search import run_latency_session_search
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.search.types import SearchType

SUPPORTED_RETRIEVERS = [
    CompletionRetriever,
    GraphCompletionRetriever,
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
    """Session facade recording exactly what a latency turn persists."""

    def __init__(self, *, available=True, auto_feedback=True):
        self.available = available
        self.auto_feedback = auto_feedback
        self.dataset_id = None
        self.qas = []
        self.entries = []
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
        for entry in self.entries:
            if entry.get("id") == entry_id:
                entry.update(merge)
                return True
        return False


@pytest.fixture
def latency(monkeypatch):
    """Latency mode with every collaborator faked except the completion path itself."""
    state = SimpleNamespace(
        manager=FakeSessionManager(),
        mode=session_search.LATENCY_OPTIMIZED,
        user=SimpleNamespace(id=uuid4()),
        structured_output=True,
        llm_calls=[],
    )

    async def fake_llm(text_input, system_prompt, response_model, **kwargs):
        state.llm_calls.append(response_model)
        if response_model is SessionMaintenanceResult:
            return SessionMaintenanceResult()
        if response_model is str:
            return "answer"
        return response_model(response="answer")

    monkeypatch.setattr(
        session_search,
        "CacheConfig",
        lambda: SimpleNamespace(session_search_mode=state.mode),
    )
    monkeypatch.setattr(session_search, "session_user", SimpleNamespace(get=lambda: state.user))
    monkeypatch.setattr(session_search, "get_session_manager", lambda: state.manager)
    monkeypatch.setattr(
        session_search.LLMGateway,
        "supports_structured_output_model",
        staticmethod(lambda model: state.structured_output),
    )
    monkeypatch.setattr(
        session_search,
        "load_latency_turn_snapshot",
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
    retriever._append_references = AsyncMock(side_effect=lambda answers, objects: answers)
    return retriever


class TestModeBoundary:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("retriever_class", SUPPORTED_RETRIEVERS)
    async def test_get_completion_answers_from_the_latency_turn(self, retriever_class, latency):
        completion = await build_retriever(retriever_class).get_completion(query="question")

        assert completion == ["answer"]
        assert len(latency.manager.qas) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("retriever_class", SUPPORTED_RETRIEVERS)
    async def test_accuracy_mode_never_enters_the_latency_turn(self, retriever_class, latency):
        latency.mode = session_search.ACCURACY_OPTIMIZED

        assert (
            await run_latency_session_search(build_retriever(retriever_class), raw_query="question")
            is None
        )
        assert latency.llm_calls == []
        assert latency.manager.qas == []

    @pytest.mark.asyncio
    async def test_get_retriever_output_maps_the_latency_result(self, latency, monkeypatch):
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
                assert "run_latency_session_search" not in inspect.getsource(method), (
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
    async def test_unsupported_call_shapes(self, call_kwargs, latency):
        result = await run_latency_session_search(
            build_retriever(CompletionRetriever), raw_query="question", **call_kwargs
        )

        assert result is None
        assert latency.manager.qas == []

    @pytest.mark.asyncio
    async def test_retriever_subclasses_are_not_supported(self, latency):
        class CustomGraphRetriever(GraphCompletionRetriever):
            pass

        assert (
            await run_latency_session_search(
                build_retriever(CustomGraphRetriever), raw_query="question"
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_missing_user_session_or_structured_output_support(self, latency):
        retriever = build_retriever(CompletionRetriever)

        latency.user = None
        assert await run_latency_session_search(retriever, raw_query="question") is None

        latency.user = SimpleNamespace(id=uuid4())
        latency.manager.available = False
        assert await run_latency_session_search(retriever, raw_query="question") is None

        latency.manager.available = True
        latency.structured_output = False
        assert await run_latency_session_search(retriever, raw_query="question") is None

        assert latency.llm_calls == []


class TestOneBlockingCompletion:
    @pytest.mark.asyncio
    async def test_turn_blocks_on_one_completion_and_defers_maintenance(self, latency, monkeypatch):
        monkeypatch.setattr(
            session_maintenance, "get_session_manager", lambda dataset_id: latency.manager
        )
        retriever = build_retriever(CompletionRetriever)

        result = await run_latency_session_search(retriever, raw_query="question")

        # The answer is back after exactly one completion; evidence is durable and queued.
        assert result.completion == ["answer"]
        assert latency.llm_calls == [session_search.get_session_search_completion_model(str)]
        assert len(latency.manager.entries) == 1
        assert latency.manager.entries[0]["status"] == "pending"
        assert session_maintenance_worker.get_tracked_evidence_ids()

        await session_maintenance_worker.drain_session_maintenance(timeout_seconds=5)

        # Only now — after the caller opted into waiting — does maintenance run.
        assert latency.llm_calls[-1] is SessionMaintenanceResult
        assert latency.manager.entries[0]["status"] == "completed"
        assert session_maintenance_worker.get_tracked_evidence_ids() == set()

    @pytest.mark.asyncio
    async def test_rapid_turns_in_one_session_are_serialized(self, latency):
        retriever = build_retriever(CompletionRetriever)
        peak = 0
        active = 0
        original = session_search.retrieve_latency_context

        async def counted(*args, **kwargs):
            nonlocal peak, active
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            try:
                return await original(*args, **kwargs)
            finally:
                active -= 1

        session_search.retrieve_latency_context = counted
        try:
            await asyncio.gather(
                run_latency_session_search(retriever, raw_query="first"),
                run_latency_session_search(retriever, raw_query="second"),
            )
        finally:
            session_search.retrieve_latency_context = original

        assert peak == 1
        assert len(latency.manager.qas) == 2

    @pytest.mark.asyncio
    async def test_auto_feedback_off_answers_without_evidence_or_maintenance(self, latency):
        latency.manager.auto_feedback = False

        result = await run_latency_session_search(
            build_retriever(CompletionRetriever), raw_query="question"
        )

        assert result.completion == ["answer"]
        assert len(latency.manager.qas) == 1
        assert latency.manager.entries == []
        assert session_maintenance_worker.get_tracked_evidence_ids() == set()


def test_session_facade_and_helpers_stay_free_of_search_mode_state():
    """Ownership boundaries the plan requires: no mode, worker, or wrapper leakage."""
    from cognee.infrastructure.session import session_manager
    from cognee.modules.retrieval.utils import references
    from cognee.modules.session_distillation import evidence

    session_manager_source = inspect.getsource(session_manager)
    assert "session_search_mode" not in session_manager_source
    assert "session_maintenance_worker" not in session_manager_source
    assert "run_latency_session_search" not in session_manager_source

    # Generic reference helpers keep receiving plain strings and lists.
    assert "session_search_models" not in inspect.getsource(references)

    # Distillation's evidence adapter owns lifecycle, never worker state.
    assert "session_maintenance_worker" not in inspect.getsource(evidence)
