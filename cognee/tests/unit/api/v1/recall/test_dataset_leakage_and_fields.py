from uuid import uuid4
import pytest
from pydantic import ValidationError
from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO
from cognee.api.v1.search.routers.get_search_router import SearchPayloadDTO
from cognee.modules.retrieval.utils.completion import generate_completion


def test_recall_payload_dto_singular_fields_mapping():
    # Test dataset_name mapping to datasets list
    payload1 = RecallPayloadDTO(query="What is eldaa?", dataset_name="eldaa")
    assert payload1.datasets == ["eldaa"]
    assert payload1.dataset_ids is None

    # Test dataset_id mapping to dataset_ids list
    test_uuid = uuid4()
    payload2 = RecallPayloadDTO(query="What is eldaa?", dataset_id=test_uuid)
    assert payload2.dataset_ids == [test_uuid]
    assert payload2.datasets is None

    # Test combining datasets lists (singular and plural) without duplicates
    payload3 = RecallPayloadDTO(query="What is eldaa?", datasets=["eldaa"], dataset_name="eldaa")
    assert payload3.datasets == ["eldaa"]

    payload4 = RecallPayloadDTO(query="What is eldaa?", datasets=["remember"], dataset_name="eldaa")
    assert set(payload4.datasets) == {"remember", "eldaa"}


def test_search_payload_dto_singular_fields_mapping():
    # Test dataset_name mapping to datasets list
    payload1 = SearchPayloadDTO(query="What is eldaa?", dataset_name="eldaa")
    assert payload1.datasets == ["eldaa"]

    # Test dataset_id mapping to dataset_ids list
    test_uuid = uuid4()
    payload2 = SearchPayloadDTO(query="What is eldaa?", dataset_id=test_uuid)
    assert payload2.dataset_ids == [test_uuid]


@pytest.mark.asyncio
async def test_generate_completion_empty_context():
    # Test generate_completion with empty and whitespace-only context
    result1 = await generate_completion(
        query="What is eldaa?",
        context="",
        user_prompt_path="context_for_question.txt",
        system_prompt_path="answer_simple_question.txt",
    )
    assert result1 == "I don't have enough context to answer this query."

    result2 = await generate_completion(
        query="What is eldaa?",
        context="   \n  ",
        user_prompt_path="context_for_question.txt",
        system_prompt_path="answer_simple_question.txt",
    )
    assert result2 == "I don't have enough context to answer this query."


@pytest.mark.asyncio
async def test_get_retriever_output_filters_by_dataset(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from cognee.modules.search.types import SearchType
    import cognee.modules.search.methods.get_retriever_output as gr_output_mod

    # Mocks
    mock_retriever = MagicMock()
    mock_retriever.prepare_session_turn_for_retrieval = AsyncMock(
        return_value=MagicMock(should_answer=True, effective_query="What is eldaa?")
    )

    class DummyChunk:
        def __init__(self, id):
            self.id = id

    matching_chunk = DummyChunk("matching-uuid")
    non_matching_chunk = DummyChunk("other-uuid")

    mock_retriever.get_retrieved_objects = AsyncMock(
        return_value=[matching_chunk, non_matching_chunk]
    )
    mock_retriever.get_context_from_objects = AsyncMock(return_value="Context from matching chunk")

    async def dummy_get_completion_from_context(context, query, **kwargs):
        return ["Completion text"]

    mock_retriever.get_completion_from_context = dummy_get_completion_from_context

    # Monkeypatch factory and engine helpers via the actual module object in sys.modules
    import sys
    import importlib
    import cognee.modules.search.methods.get_retriever_output

    gr_mod = sys.modules["cognee.modules.search.methods.get_retriever_output"]
    importlib.reload(gr_mod)

    monkeypatch.setattr(
        gr_mod, "get_search_type_retriever_instance", AsyncMock(return_value=mock_retriever)
    )

    monkeypatch.setattr(gr_mod, "get_graph_engine", AsyncMock())

    monkeypatch.setattr(gr_mod, "update_node_access_timestamps", AsyncMock())

    # Mock submodules in sys.modules using monkeypatch.setitem to guarantee they are resolved correctly during runtime imports
    import sys
    from unittest.mock import MagicMock

    mock_graph_module = MagicMock()
    mock_graph_module.get_dataset_node_ids = AsyncMock(return_value={"matching-uuid"})
    monkeypatch.setitem(
        sys.modules, "cognee.modules.graph.methods.get_dataset_node_ids", mock_graph_module
    )
    if "cognee.modules.graph.methods" in sys.modules:
        monkeypatch.setattr(
            sys.modules["cognee.modules.graph.methods"],
            "get_dataset_node_ids",
            mock_graph_module.get_dataset_node_ids,
        )

    mock_dataset = MagicMock()
    mock_dataset.id = uuid4()
    mock_data_module = MagicMock()
    mock_data_module.get_authorized_existing_datasets = AsyncMock(return_value=[mock_dataset])
    monkeypatch.setitem(
        sys.modules,
        "cognee.modules.data.methods.get_authorized_existing_datasets",
        mock_data_module,
    )
    if "cognee.modules.data.methods" in sys.modules:
        monkeypatch.setattr(
            sys.modules["cognee.modules.data.methods"],
            "get_authorized_existing_datasets",
            mock_data_module.get_authorized_existing_datasets,
        )

    result = await gr_mod.get_retriever_output(
        query_type=SearchType.CHUNKS, query_text="What is eldaa?", user=MagicMock()
    )

    # Verify that only the matching chunk is retained
    assert result.result_object == [matching_chunk]
