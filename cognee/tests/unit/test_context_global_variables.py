from uuid import uuid4

import pytest

from cognee.context_global_variables import (
    current_dataset_id,
    set_database_global_context_variables,
)


@pytest.mark.asyncio
async def test_database_context_sets_and_resets_current_dataset_id(monkeypatch):
    dataset_id = uuid4()
    user_id = uuid4()
    current_dataset_id.set("outer")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    async with set_database_global_context_variables(dataset_id, user_id):
        assert current_dataset_id.get() == str(dataset_id)

    assert current_dataset_id.get() == "outer"
