import os
import asyncio
import pytest
from unittest.mock import patch
from cognee.api.v1.health.health import HealthChecker, HealthStatus
from cognee.base_config import get_base_config


@pytest.mark.asyncio
async def test_health_check_does_not_delete_existing_file():
    """Test that check_file_storage does not overwrite or delete an existing health_check_test file."""
    config = get_base_config()
    test_dir = config.data_root_directory
    os.makedirs(test_dir, exist_ok=True)

    # Create a dummy file that simulates existing user data that happens to be named "health_check_test"
    existing_file = os.path.join(test_dir, "health_check_test")
    with open(existing_file, "w") as f:
        f.write("important_user_data")

    checker = HealthChecker()

    # Run the check
    result = await checker.check_file_storage()

    # The check should be healthy
    assert result.status == HealthStatus.HEALTHY

    # The existing file should still exist and its content should be untouched
    assert os.path.exists(existing_file), "The existing file was deleted by the health check!"
    with open(existing_file, "r") as f:
        content = f.read()
    assert content == "important_user_data", (
        "The existing file was overwritten by the health check!"
    )

    # Clean up
    os.remove(existing_file)
