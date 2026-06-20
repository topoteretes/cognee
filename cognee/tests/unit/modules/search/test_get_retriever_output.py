import importlib
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.search.methods.get_retriever_output import (
    _count_retrieved_objects,
    _normalize_completion_value,
    get_retriever_output,
)
from cognee.modules.search.types import SearchType

# Resolve the module object explicitly. The package __init__ re-exports the
# `get_retriever_output` function under the same name as this submodule, so a
# dotted-string patch target ("...methods.get_retriever_output.<attr>") resolves
# to the function rather than the module and fails. Patching the module object
# directly via patch.object is order-independent and reliable.
get_retriever_output_module = importlib.import_module(
    "cognee.modules.search.methods.get_retriever_output"
)


class _FakeGraphEngine:
    async def is_empty(self):
        return False


class _EffectiveQueryRetriever:
    def __init__(self):
        self.preparation = SessionTurnPreparation(
            should_answer=True,
            effective_query="What should I audit in Lisbon?",
        )
        self.retrieved_query = None
        self.context_query = None
        self.completion_kwargs = None

    async def prepare_session_turn_for_retrieval(self, query):
        assert query == "That was wrong. What should I audit in Lisbon?"
        return self.preparation

    async def get_retrieved_objects(self, query):
        self.retrieved_query = query
        return [{"id": "obj-1"}]

    async def get_context_from_objects(self, query, retrieved_objects):
        self.context_query = query
        assert retrieved_objects == [{"id": "obj-1"}]
        return "context"

    async def get_completion_from_context(
        self,
        query,
        retrieved_objects,
        context,
        effective_query=None,
        turn_preparation=None,
    ):
        self.completion_kwargs = {
            "query": query,
            "effective_query": effective_query,
            "turn_preparation": turn_preparation,
            "context": context,
        }
        return ["answer"]


class _NoAnswerRetriever:
    async def prepare_session_turn_for_retrieval(self, query):
        return SessionTurnPreparation(
            should_answer=False,
            response_to_user="Thanks, I noted that.",
            effective_query=query,
        )

    async def get_retrieved_objects(self, query):
        raise AssertionError("retrieval should be skipped")


class SampleAnswer(BaseModel):
    answer: str
    confidence: float


class _StructuredCompletionRetriever:
    async def prepare_session_turn_for_retrieval(self, query):
        return SessionTurnPreparation(should_answer=True, effective_query=query)

    async def get_retrieved_objects(self, query):
        return []

    async def get_context_from_objects(self, query, retrieved_objects):
        return "context"

    async def get_completion_from_context(self, query, retrieved_objects, context):
        return SampleAnswer(answer="structured", confidence=0.75)


@pytest.mark.asyncio
async def test_get_retriever_output_uses_effective_query_before_retrieval():
    retriever = _EffectiveQueryRetriever()
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=retriever,
        ),
    ):
        result = await get_retriever_output(
            SearchType.CHUNKS,
            "That was wrong. What should I audit in Lisbon?",
        )

    assert retriever.retrieved_query == "What should I audit in Lisbon?"
    assert retriever.context_query == "What should I audit in Lisbon?"
    assert retriever.completion_kwargs == {
        "query": "That was wrong. What should I audit in Lisbon?",
        "effective_query": "What should I audit in Lisbon?",
        "turn_preparation": retriever.preparation,
        "context": "context",
    }
    assert result.completion == ["answer"]


@pytest.mark.asyncio
async def test_get_retriever_output_skips_retrieval_for_no_answer_turn():
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=_NoAnswerRetriever(),
        ),
    ):
        result = await get_retriever_output(SearchType.CHUNKS, "That was wrong.")

    assert result.result_object is None
    assert result.context is None
    assert result.completion == ["Thanks, I noted that."]


@pytest.mark.asyncio
async def test_get_retriever_output_normalizes_structured_response_model_completion():
    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch.object(
            get_retriever_output_module,
            "get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=_StructuredCompletionRetriever(),
        ),
    ):
        result = await get_retriever_output(SearchType.GRAPH_COMPLETION, "structured query")

    assert result.completion == {"answer": "structured", "confidence": 0.75}


def test_count_retrieved_objects_counts_structured_lists():
    assert _count_retrieved_objects({"chunks": [1, 2], "entities": [3]}) == 3


def test_count_retrieved_objects_preserves_existing_shapes():
    assert _count_retrieved_objects(None) == 0
    assert _count_retrieved_objects(["a", "b"]) == 2
    assert _count_retrieved_objects({"triplets": []}) == 0
    assert _count_retrieved_objects({"metadata": "value"}) == 1
    assert _count_retrieved_objects("answer") == 1


def test_normalize_completion_value_serializes_base_model():
    model = SampleAnswer(answer="42", confidence=0.9)
    assert _normalize_completion_value(model) == {
        "answer": "42",
        "confidence": 0.9,
    }


def test_normalize_completion_value_handles_nested_structures():
    model = SampleAnswer(answer="yes", confidence=1.0)
    assert _normalize_completion_value([model, "plain"]) == [
        {"answer": "yes", "confidence": 1.0},
        "plain",
    ]


@pytest.mark.asyncio
async def test_search_result_payload_accepts_normalized_response_model_completion():
    """SearchResultPayload must accept structured completions after normalization (#3048)."""
    from cognee.modules.search.models.SearchResultPayload import SearchResultPayload

    model = SampleAnswer(answer="structured", confidence=0.75)
    payload = SearchResultPayload(
        result_object=None,
        context=None,
        completion=_normalize_completion_value(model),
        search_type=SearchType.GRAPH_COMPLETION,
    )

    assert payload.completion == {"answer": "structured", "confidence": 0.75}
