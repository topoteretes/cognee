import pytest
import uuid
from cognee.tasks.cleanup import cleanup_unused_data


@pytest.mark.asyncio
async def test_cleanup_dry_run():
    result = await cleanup_unused_data(30, dry_run=True)
    assert result["status"] == "dry-run"
    assert "unused_data_count" in result
    assert isinstance(result["unused_data_count"], int)
    assert result["unused_data_count"] >= 0
    assert "cutoff_date" in result


@pytest.mark.asyncio
async def test_cleanup_execute():
    result = await cleanup_unused_data(30, dry_run=False)
    assert result["status"] == "executed"
    assert "deleted_count" in result
    assert isinstance(result["deleted_count"], int)
    assert result["deleted_count"] >= 0
    assert "cutoff_date" in result


@pytest.mark.asyncio
async def test_invalid_days_threshold():
    with pytest.raises(ValueError, match="days_threshold must be an int >= 0"):
        await cleanup_unused_data(-1)


@pytest.mark.asyncio
async def test_cleanup_with_user_id():
    uid = uuid.uuid4()
    res = await cleanup_unused_data(dry_run=True, user_id=uid)
    assert res["status"] == "dry-run"
    assert res["user_id"] == str(uid)
