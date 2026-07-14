"""Tests for the MemoryAdapter (remember/recall/forget + ID mapping)."""

from __future__ import annotations

from uuid import UUID

import pytest

from config import BotConfig
from memory_adapter import MemoryAdapter


@pytest.fixture
def adapter():
    """Create a MemoryAdapter with test config."""
    config = BotConfig(
        dataset_name="test_support_threads",
        memory_scope="channel",
        top_k=5,
        min_relevance_score=0.0,
    )
    return MemoryAdapter(config)


class TestIngestResolvedThread:
    """Tests for ingest_resolved_thread()."""

    @pytest.mark.asyncio
    async def test_ingest_calls_remember_with_correct_params(
        self, adapter, sample_resolved_threads, mock_cognee
    ):
        """Verify cognee.remember() is called with correct dataset_name and session_id."""
        thread = sample_resolved_threads[0]
        await adapter.ingest_resolved_thread(thread)

        mock_cognee["remember"].assert_called_once()
        call_args = mock_cognee["remember"].call_args
        assert call_args[0][1] == "test_support_threads"  # dataset_name
        assert call_args[1]["session_id"] == "channel_support"  # session_id

    @pytest.mark.asyncio
    async def test_ingest_persists_thread_data_id_mapping(
        self, adapter, sample_resolved_threads, mock_cognee
    ):
        """Verify the adapter persists the thread_ts ↔ data_id mapping after ingestion."""
        thread = sample_resolved_threads[0]
        result = await adapter.ingest_resolved_thread(thread)

        # The mock returns items with an "id" field
        assert adapter.has_mapping(thread.thread_id)
        data_id = adapter.get_data_id(thread.thread_id)
        assert data_id is not None
        assert isinstance(data_id, UUID)


class TestFindSimilarIssues:
    """Tests for find_similar_issues()."""

    @pytest.mark.asyncio
    async def test_returns_ranked_citations(
        self, adapter, mock_cognee
    ):
        """Mock recall() with results → assert ranked Citation list."""
        citations = await adapter.find_similar_issues("auth timeout", "support")

        assert len(citations) > 0
        # Verify recall was called with correct params
        mock_cognee["recall"].assert_called_once()
        call_kwargs = mock_cognee["recall"].call_args[1]
        assert call_kwargs["datasets"] == ["test_support_threads"]
        assert call_kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_empty_when_no_matches(
        self, adapter, mock_cognee_empty
    ):
        """Mock recall() returning [] → assert empty citations."""
        citations = await adapter.find_similar_issues("irrelevant query", "support")
        assert citations == []


class TestForgetThread:
    """Tests for forget_thread()."""

    @pytest.mark.asyncio
    async def test_forget_looks_up_data_id(
        self, adapter, sample_resolved_threads, mock_cognee
    ):
        """Verify the adapter retrieves the correct UUID before calling forget()."""
        # First ingest to create the mapping
        thread = sample_resolved_threads[0]
        await adapter.ingest_resolved_thread(thread)
        data_id = adapter.get_data_id(thread.thread_id)

        # Now forget
        await adapter.forget_thread(thread.thread_id)

        mock_cognee["forget"].assert_called_once_with(
            data_id=data_id,
            dataset="test_support_threads",
        )

    @pytest.mark.asyncio
    async def test_forget_unknown_id_raises_error(
        self, adapter, mock_cognee
    ):
        """Verify clear error when thread was never ingested."""
        with pytest.raises(KeyError, match="was never ingested"):
            await adapter.forget_thread("NEVER_INGESTED")

    @pytest.mark.asyncio
    async def test_forget_removes_mapping(
        self, adapter, sample_resolved_threads, mock_cognee
    ):
        """After forget, the mapping should be removed."""
        thread = sample_resolved_threads[0]
        await adapter.ingest_resolved_thread(thread)
        assert adapter.has_mapping(thread.thread_id)

        await adapter.forget_thread(thread.thread_id)
        assert not adapter.has_mapping(thread.thread_id)


class TestForgetEverything:
    """Tests for forget_all()."""

    @pytest.mark.asyncio
    async def test_forget_everything(self, adapter, mock_cognee):
        """Verify cognee.forget(everything=True) is called and mappings cleared."""
        await adapter.forget_all()

        mock_cognee["forget"].assert_called_once_with(everything=True)
