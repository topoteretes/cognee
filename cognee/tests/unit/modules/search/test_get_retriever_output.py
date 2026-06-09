from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.types import SearchType


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


@pytest.mark.asyncio
async def test_get_retriever_output_uses_effective_query_before_retrieval():
    retriever = _EffectiveQueryRetriever()
    with (
        patch(
            "cognee.modules.search.methods.get_retriever_output.get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch(
            "cognee.modules.search.methods.get_retriever_output.get_search_type_retriever_instance",
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
        patch(
            "cognee.modules.search.methods.get_retriever_output.get_graph_engine",
            new_callable=AsyncMock,
            return_value=_FakeGraphEngine(),
        ),
        patch(
            "cognee.modules.search.methods.get_retriever_output.get_search_type_retriever_instance",
            new_callable=AsyncMock,
            return_value=_NoAnswerRetriever(),
        ),
    ):
        result = await get_retriever_output(SearchType.CHUNKS, "That was wrong.")

    assert result.result_object is None
    assert result.context is None
    assert result.completion == ["Thanks, I noted that."]
