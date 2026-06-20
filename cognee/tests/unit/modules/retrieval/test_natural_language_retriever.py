import pytest
import json
from unittest.mock import AsyncMock, patch
from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever

@pytest.mark.asyncio
async def test_natural_language_retriever_get_context():
    retriever = NaturalLanguageRetriever()
    
    # Test with empty retrieved objects
    assert await retriever.get_context_from_objects("query", None) is None
    assert await retriever.get_context_from_objects("query", []) is None

    # Test with valid retrieved objects
    objects = [{"node_id": "1", "name": "Node1"}, {"node_id": "2", "name": "Node2"}]
    context = await retriever.get_context_from_objects("query", objects)
    assert context is not None
    loaded = json.loads(context)
    assert loaded == objects

@pytest.mark.asyncio
async def test_natural_language_retriever_get_completion():
    retriever = NaturalLanguageRetriever()

    # Test with empty context
    assert await retriever.get_completion_from_context("query", None, None) is None
    assert await retriever.get_completion_from_context("query", None, "") is None

    # Test with valid context and mock generate_completion
    with patch(
        "cognee.modules.retrieval.natural_language_retriever.generate_completion",
        new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = "Completion result"
        completion = await retriever.get_completion_from_context("query", None, "some context")
        
        assert completion == ["Completion result"]
        mock_generate.assert_awaited_once_with(
            query="query",
            context="some context",
            user_prompt_path="context_for_question.txt",
            system_prompt_path="answer_simple_question.txt"
        )
