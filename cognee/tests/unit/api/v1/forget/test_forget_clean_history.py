"""Tests for the clean_history option in cognee.forget().

clean_history=True deletes the caller's own search history (every row this
user owns) before the data is deleted; the default keeps it. The flag is
forwarded to a remote instance when connected through serve().
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

forget_module = importlib.import_module("cognee.api.v1.forget.forget")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
low_level_module = importlib.import_module("cognee.low_level")
search_operations_module = importlib.import_module("cognee.modules.search.operations")

USER = SimpleNamespace(id=uuid4())
DATASET_ID = uuid4()
DATA_ID = uuid4()


def _local(**overrides):
    """Patch forget() to run locally with the data paths replaced by recording mocks."""
    calls = []

    def record(name, result):
        calls.append(name)
        return result

    purge = AsyncMock(
        side_effect=lambda user_id: record(
            ("history", user_id), {"queries_deleted": 2, "results_deleted": 3}
        )
    )
    paths = {
        "_forget_data_item": AsyncMock(
            side_effect=lambda *a: record("item", {"status": "success"})
        ),
        "_forget_data_memory": AsyncMock(
            side_effect=lambda *a: record("item_memory", {"status": "success"})
        ),
        "_forget_dataset": AsyncMock(
            side_effect=lambda *a: record("dataset", {"status": "success"})
        ),
        "_forget_everything": AsyncMock(
            side_effect=lambda *a: record("everything", {"status": "success"})
        ),
    }
    patches = [
        patch.object(low_level_module, "setup", AsyncMock()),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(forget_module, "_resolve_dataset_id", AsyncMock(return_value=DATASET_ID)),
        patch.object(search_operations_module, "delete_user_search_history", purge),
    ] + [patch.object(forget_module, name, mock) for name, mock in paths.items()]
    return patches, purge, calls


@pytest.mark.asyncio
async def test_item_forget_keeps_history_by_default():
    patches, purge, calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        result = await forget_module.forget(data_id=DATA_ID, dataset_id=DATASET_ID, user=USER)

    purge.assert_not_awaited()
    assert calls == ["item"]
    assert result["history_cleaned"] is False


@pytest.mark.asyncio
async def test_clean_history_purges_the_callers_rows_before_the_data():
    patches, purge, calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        result = await forget_module.forget(
            data_id=DATA_ID, dataset_id=DATASET_ID, clean_history=True, user=USER
        )

    purge.assert_awaited_once_with(USER.id)
    assert calls == [("history", USER.id), "item"]
    assert result["history_cleaned"] is True


@pytest.mark.asyncio
async def test_clean_history_applies_to_memory_only_forgets():
    patches, purge, calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        result = await forget_module.forget(
            data_id=DATA_ID, dataset_id=DATASET_ID, memory_only=True, clean_history=True, user=USER
        )

    purge.assert_awaited_once_with(USER.id)
    assert calls == [("history", USER.id), "item_memory"]
    assert result["history_cleaned"] is True


@pytest.mark.asyncio
async def test_everything_forget_does_not_purge_twice():
    """_forget_everything removes the caller's history itself."""
    patches, purge, calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        await forget_module.forget(everything=True, clean_history=True, user=USER)

    purge.assert_not_awaited()
    assert calls == ["everything"]


@pytest.mark.asyncio
async def test_clean_history_is_forwarded_to_the_remote_client():
    remote_client = SimpleNamespace(forget=AsyncMock(return_value={"status": "success"}))

    with patch.object(serve_state_module, "get_remote_client", return_value=remote_client):
        await forget_module.forget(
            data_id=DATA_ID, dataset_id=DATASET_ID, clean_history=True, user=USER
        )

    assert remote_client.forget.call_args.kwargs["clean_history"] is True


@pytest.mark.asyncio
async def test_clean_history_alone_deletes_only_the_history():
    patches, purge, calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        result = await forget_module.forget(clean_history=True, user=USER)

    purge.assert_awaited_once_with(USER.id)
    assert calls == [("history", USER.id)]
    assert result == {"history_cleaned": True, "status": "success"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"memory_only": True}, {"data_id": DATA_ID}],
    ids=["memory_only_without_dataset", "data_id_without_dataset"],
)
async def test_invalid_target_raises_before_the_history_is_touched(kwargs):
    patches, purge, calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        pytest.raises(ValueError),
    ):
        await forget_module.forget(clean_history=True, user=USER, **kwargs)

    purge.assert_not_awaited()
    assert calls == []


@pytest.mark.asyncio
async def test_no_target_and_no_clean_history_still_raises():
    patches, purge, _calls = _local()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        pytest.raises(ValueError, match="clean_history=True"),
    ):
        await forget_module.forget(user=USER)

    purge.assert_not_awaited()
