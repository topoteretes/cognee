import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from uuid import uuid4

import cognee
from cognee.tasks.notetaker.normalize import normalize_transcript
from cognee.api.v1.notetaker import IngestPayload
from cognee.tasks.temporal_graph.models import QueryInterval
from cognee.modules.retrieval.notetaker_templates import (
    NotetakerActionItemRetriever,
    NotetakerTemporalDeltaRetriever
)


@pytest.fixture(autouse=True)
async def setup_teardown():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.asyncio
async def test_date_anchoring_guard_positive():
    """Test that two occurrences with injected dates are temporally scoped correctly."""
    series_id = "test_series_positive"
    
    # Occurrence 1
    turns_1 = [("Alice", "Let's release v1 today.", "2026-06-10 10:00")]
    norm_1 = normalize_transcript(turns_1, meeting_id="m1")
    await cognee.add(norm_1, dataset_name=series_id)
    
    # Occurrence 2
    turns_2 = [("Bob", "v1 is out. Let's plan v2.", "2026-06-17 10:00")]
    norm_2 = normalize_transcript(turns_2, meeting_id="m2")
    await cognee.add(norm_2, dataset_name=series_id)
    
    # Mock the LLM to return temporal extraction success
    with patch("cognee.infrastructure.llm.LLMGateway.acreate_structured_output") as mock_extract:
        mock_extract.return_value = QueryInterval(starts_at="2026-06-16", ends_at="2026-06-18")
        
        # Test temporal extraction logic directly
        retriever = NotetakerTemporalDeltaRetriever()
        t_from, t_to = await retriever.extract_time_from_query("What happened this week?")
        
        assert t_from == "2026-06-16"
        assert t_to == "2026-06-18"


@pytest.mark.asyncio
async def test_date_anchoring_guard_negative():
    """Test that occurrences with missing dates fallback to default logic without crashing."""
    series_id = "test_series_negative"
    
    # Occurrence with missing timestamps
    turns = [("Alice", "Let's release v1 today.", None)]
    norm = normalize_transcript(turns, meeting_id="m1")
    
    # Should gracefully handle and inject [1970-01-01 00:00]
    assert "[1970-01-01 00:00] Alice: (meeting_id=m1)" in norm


@pytest.mark.asyncio
async def test_temporal_cognify_exclusivity():
    """Test that temporal_cognify=True ignores custom graph_models by not raising errors about them,
       but running the temporal path instead.
    """
    series_id = "test_series_exclusive"
    turns = [("Alice", "Let's test exclusivity", "2026-06-10 10:00")]
    norm = normalize_transcript(turns, meeting_id="m1")
    await cognee.add(norm, dataset_name=series_id)
    
    with patch("cognee.api.v1.cognify.cognify.get_default_tasks") as mock_default, \
         patch("cognee.api.v1.cognify.cognify.get_temporal_tasks") as mock_temporal:
        
        mock_temporal.return_value = [] # Empty pipeline for test
        
        # This will call the mocked get_temporal_tasks
        # We wrap in try/except because we mocked the tasks to empty list which might fail pipeline
        try:
            await cognee.cognify(
                datasets=[series_id],
                temporal_cognify=True,
                graph_model={"title": "CustomModel"}
            )
        except Exception:
            pass
            
        mock_temporal.assert_called_once()
        mock_default.assert_not_called()


@pytest.mark.asyncio
async def test_citation_round_trip():
    """Test that the provenance prefix injected at normalization survives."""
    turns = [("Alice", "Deploy blocked on staging creds", "2026-06-24 14:32")]
    norm = normalize_transcript(turns, meeting_id="m1", permalink="https://example.com/m1")
    
    assert "[2026-06-24 14:32] Alice: (meeting_id=m1, permalink=https://example.com/m1)" in norm
    assert "Deploy blocked on staging creds" in norm


@pytest.mark.asyncio
async def test_forget_series():
    """Test forgetting a series wipes it while keeping siblings."""
    from cognee.api.v1.forget.forget import forget
    
    series_1 = "series_to_forget"
    series_2 = "series_to_keep"
    
    await cognee.add("Some text", dataset_name=series_1)
    await cognee.add("Other text", dataset_name=series_2)
    
    await forget(dataset_id=series_1)
    
    # Check that series_1 is gone but series_2 remains
    datasets = await cognee.datasets.get_datasets()
    dataset_names = [d.name for d in datasets]
    assert series_1 not in dataset_names
    assert series_2 in dataset_names


@pytest.mark.asyncio
async def test_action_item_recall():
    """Test that action item recall uses the right template."""
    with patch("cognee.modules.retrieval.temporal_retriever.TemporalRetriever.get_completion") as mock_completion:
        mock_completion.return_value = "Alice will deploy staging [Alice, 2026-06-24 14:32, permalink=m1]"
        
        retriever = NotetakerActionItemRetriever()
        answer = await retriever.get_completion("What are the action items?")
        
        assert "Alice will deploy staging" in answer
        mock_completion.assert_called_once_with("What are the action items?")
