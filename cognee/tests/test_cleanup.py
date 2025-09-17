import pytest
import uuid
from datetime import datetime, timezone
from cognee.tasks.cleanup import cleanup_unused_data


@pytest.mark.asyncio
async def test_cleanup_dry_run():
    result = await cleanup_unused_data(30, dry_run=True)
    assert result["status"] == "dry-run"
    assert "unused_data_count" in result
    assert "cutoff_date" in result

    # Ensure ISO-8601 with timezone
    dt = datetime.fromisoformat(result["cutoff_date"])
    assert dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) == timezone.utc.utcoffset(dt)


@pytest.mark.asyncio
async def test_cleanup_execute_mode():
    dry = await cleanup_unused_data(30, dry_run=True)
    exe = await cleanup_unused_data(30, dry_run=False)

    assert exe["status"] == "executed"
    assert "deleted_count" in exe
    assert isinstance(exe["deleted_count"], int)
    assert exe["deleted_count"] >= 0
    assert exe["deleted_count"] == dry["unused_data_count"]


@pytest.mark.asyncio
async def test_cleanup_negative_days_threshold():
    with pytest.raises(ValueError, match="days_threshold must be an int >= 0"):
        await cleanup_unused_data(-5)


@pytest.mark.asyncio
async def test_cleanup_defaults_and_user_scope():
    # default threshold via kwargs only
    res = await cleanup_unused_data(dry_run=True, user_id=uuid.uuid4())
    assert res["status"] == "dry-run"
    assert "cutoff_date" in res
    assert res["user_id"] is not None
