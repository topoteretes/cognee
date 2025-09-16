import pytest
import asyncio
from cognee.tasks.cleanup import cleanup_unused_data

@pytest.mark.asyncio
async def test_cleanup_dry_run():
    result = await cleanup_unused_data(days_threshold=1, dry_run=True)
    assert result["status"] == "dry_run"
    assert "unused_count" in result

@pytest.mark.asyncio
async def test_cleanup_execute():
    result = await cleanup_unused_data(days_threshold=1, dry_run=False)
    assert result["status"] == "completed"
    assert "deleted_count" in result
