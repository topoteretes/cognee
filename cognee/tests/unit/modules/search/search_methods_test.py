import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pylint.checkers.utils import node_type

from cognee.exceptions import InvalidValueError
from cognee.modules.search.methods.search import search, specific_search
from cognee.modules.search.types import SearchType
from cognee.modules.users.models import User
import sys

search_module = sys.modules.get("cognee.modules.search.methods.search")


@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    return user


@pytest.mark.asyncio
@patch.object(search_module, "log_query")
@patch.object(search_module, "log_result")
@patch.object(search_module, "specific_search")
async def test_search(
    mock_specific_search,
    mock_log_result,
    mock_log_query,
    mock_user,
):
    # Setup
    query_text = "test query"
    query_type = SearchType.CHUNKS
    datasets = ["dataset1", "dataset2"]

    # Mock the query logging
    mock_query = MagicMock()
    mock_query.id = uuid.uuid4()
    mock_log_query.return_value = mock_query

    # Mock document IDs
    doc_id1 = uuid.uuid4()
    doc_id2 = uuid.uuid4()

    # Mock search results
    search_results = [
        {"document_id": str(doc_id1), "content": "Result 1"},
        {"document_id": str(doc_id2), "content": "Result 2"},
    ]
    mock_specific_search.return_value = search_results

    # Execute
    await search(query_text, query_type, datasets, mock_user)

    # Verify
    mock_log_query.assert_called_once_with(query_text, query_type.value, mock_user.id)
    mock_specific_search.assert_called_once_with(
        query_type,
        query_text,
        mock_user,
        system_prompt_path="answer_simple_question.txt",
        top_k=10,
        node_type=None,
        node_name=None,
    )

    # Verify result logging
    mock_log_result.assert_called_once()
    # Check that the first argument is the query ID
    assert mock_log_result.call_args[0][0] == mock_query.id
    # The second argument should be the JSON string of the filtered results
    # We can't directly compare the JSON strings due to potential ordering differences
    # So we parse the JSON and compare the objects
    logged_results = json.loads(mock_log_result.call_args[0][1])
    assert len(logged_results) == 2
    assert logged_results[0]["document_id"] == str(doc_id1)
    assert logged_results[1]["document_id"] == str(doc_id2)


@pytest.mark.asyncio
@patch.object(search_module, "SummariesRetriever")
@patch.object(search_module, "send_telemetry")
async def test_specific_search_summaries(mock_send_telemetry, mock_summaries_retriever, mock_user):
    # Setup
    query = "test query"
    query_type = SearchType.SUMMARIES

    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_completion = AsyncMock()
    mock_retriever.get_completion.return_value = [{"content": "Summary result"}]
    mock_summaries_retriever.return_value = mock_retriever

    # Execute
    results = await specific_search(query_type, query, mock_user)

    # Verify
    mock_summaries_retriever.assert_called_once()
    mock_retriever.get_completion.assert_called_once_with(query)
    mock_send_telemetry.assert_called()
    assert len(results) == 1
    assert results[0]["content"] == "Summary result"


@pytest.mark.asyncio
@patch.object(search_module, "InsightsRetriever")
@patch.object(search_module, "send_telemetry")
async def test_specific_search_insights(mock_send_telemetry, mock_insights_retriever, mock_user):
    # Setup
    query = "test query"
    query_type = SearchType.INSIGHTS

    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_completion = AsyncMock()
    mock_retriever.get_completion.return_value = [{"content": "Insight result"}]
    mock_insights_retriever.return_value = mock_retriever

    # Execute
    results = await specific_search(query_type, query, mock_user)

    # Verify
    mock_insights_retriever.assert_called_once()
    mock_retriever.get_completion.assert_called_once_with(query)
    mock_send_telemetry.assert_called()
    assert len(results) == 1
    assert results[0]["content"] == "Insight result"


@pytest.mark.asyncio
@patch.object(search_module, "ChunksRetriever")
@patch.object(search_module, "send_telemetry")
async def test_specific_search_chunks(mock_send_telemetry, mock_chunks_retriever, mock_user):
    # Setup
    query = "test query"
    query_type = SearchType.CHUNKS

    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_completion = AsyncMock()
    mock_retriever.get_completion.return_value = [{"content": "Chunk result"}]
    mock_chunks_retriever.return_value = mock_retriever

    # Execute
    results = await specific_search(query_type, query, mock_user)

    # Verify
    mock_chunks_retriever.assert_called_once()
    mock_retriever.get_completion.assert_called_once_with(query)
    mock_send_telemetry.assert_called()
    assert len(results) == 1
    assert results[0]["content"] == "Chunk result"


@pytest.mark.asyncio
async def test_specific_search_invalid_type(mock_user):
    # Setup
    query = "test query"
    query_type = "INVALID_TYPE"  # Not a valid SearchType

    # Execute and verify
    with pytest.raises(InvalidValueError) as excinfo:
        await specific_search(query_type, query, mock_user)

    assert "Unsupported search type" in str(excinfo.value)
