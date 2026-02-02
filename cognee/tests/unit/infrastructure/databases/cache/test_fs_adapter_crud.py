"""Unit tests for FsCacheAdapter CRUD operations."""

import tempfile
import pytest
from unittest.mock import patch

from cognee.infrastructure.databases.exceptions import SessionQAEntryValidationError


@pytest.fixture
def adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch(
            "cognee.infrastructure.databases.cache.fscache.FsCacheAdapter.get_storage_config",
            return_value={"data_root_directory": tmpdir},
        ):
            from cognee.infrastructure.databases.cache.fscache.FsCacheAdapter import (
                FSCacheAdapter,
            )

            yield FSCacheAdapter()


@pytest.mark.asyncio
async def test_create_and_get(adapter):
    """Create a QA entry and retrieve it."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    entries = await adapter.get_all_qa_entries("u1", "s1")
    assert len(entries) == 1 and entries[0]["qa_id"] == "id1"


@pytest.mark.asyncio
async def test_update(adapter):
    """Update a QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    ok = await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=5)
    assert ok and (await adapter.get_all_qa_entries("u1", "s1"))[0]["feedback_score"] == 5


@pytest.mark.asyncio
async def test_update_invalid_raises(adapter):
    """Raise SessionQAEntryValidationError when feedback_score is out of range or gets wrong format."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    with pytest.raises(SessionQAEntryValidationError):
        await adapter.update_qa_entry("u1", "s1", "id1", feedback_score=10)

    with pytest.raises(SessionQAEntryValidationError):
        await adapter.update_qa_entry("u1", "s1", "id1", feedback_text=5)


@pytest.mark.asyncio
async def test_delete_entry(adapter):
    """Delete a single QA entry by qa_id."""
    await adapter.create_qa_entry("u1", "s1", "Q1", "C1", "A1", qa_id="id1")
    await adapter.create_qa_entry("u1", "s1", "Q2", "C2", "A2", qa_id="id2")
    ok = await adapter.delete_qa_entry("u1", "s1", "id1")
    assert ok and len(await adapter.get_all_qa_entries("u1", "s1")) == 1


@pytest.mark.asyncio
async def test_delete_session(adapter):
    """Delete the entire session and all its entries."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    ok = await adapter.delete_session("u1", "s1")
    assert ok and await adapter.get_all_qa_entries("u1", "s1") == []


@pytest.mark.asyncio
async def test_prune(adapter):
    """Flush all cached data."""
    await adapter.create_qa_entry("u1", "s1", "Q", "C", "A", qa_id="id1")
    await adapter.prune()
    assert await adapter.get_all_qa_entries("u1", "s1") == []
