import pytest
from cognee.tasks.cleanup import cleanup_unused_data


@pytest.mark.asyncio
async def test_cleanup_dry_run():
    result = await cleanup_unused_data(30, dry_run=True)
    assert result["status"] == "dry-run"
    assert "unused_data_count" in result
    assert "cutoff_date" in result

@pytest.mark.asyncio
async def test_cleanup_execute():
    result = await cleanup_unused_data(30, dry_run=False)
    assert result["status"] == "executed"
    assert "deleted_count" in result
    assert "cutoff_date" in result

@pytest.mark.asyncio
async def test_cleanup_negative_days_threshold():
    with pytest.raises(ValueError, match="days_threshold cannot be negative"):
        await cleanup_unused_data(-5)
