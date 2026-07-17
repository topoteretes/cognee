"""Regression tests for deleting data while the dataset's pipeline is running.

Deleting a data item while the dataset's cognify pipeline is in
``DATASET_PROCESSING_STARTED`` used to race the in-flight pipeline and surface
as an opaque 500. ``datasets.delete_data`` (the chokepoint shared by
``POST /v1/forget`` with ``dataId`` and
``DELETE /v1/datasets/{dataset_id}/data/{data_id}``) must instead raise
``DatasetProcessingInProgressError`` (HTTP 409) so clients know to retry once
processing completes.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.datasets.datasets import datasets
from cognee.modules.data.exceptions import DatasetProcessingInProgressError
from cognee.modules.pipelines.models import PipelineRunStatus


def _make_dataset(dataset_id, owner_id):
    dataset = MagicMock()
    dataset.id = dataset_id
    dataset.owner_id = owner_id
    return dataset


def _make_user(user_id):
    user = MagicMock()
    user.id = user_id
    return user


@pytest.mark.asyncio
async def test_delete_data_raises_409_while_dataset_is_processing():
    """delete_data must fail fast with a 409-mapped error during processing."""
    dataset_id = uuid4()
    user = _make_user(uuid4())
    dataset = _make_dataset(dataset_id, user.id)

    with (
        patch(
            "cognee.api.v1.datasets.datasets.get_authorized_dataset",
            new=AsyncMock(return_value=dataset),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.get_pipeline_status",
            new=AsyncMock(
                return_value={str(dataset_id): PipelineRunStatus.DATASET_PROCESSING_STARTED}
            ),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.get_dataset_data",
            new=AsyncMock(return_value=[]),
        ) as mock_get_dataset_data,
    ):
        with pytest.raises(DatasetProcessingInProgressError) as exc_info:
            await datasets.delete_data(dataset_id, uuid4(), user)

    # 409 lets clients distinguish "retry later" from a genuine server fault.
    assert exc_info.value.status_code == 409
    # The guard must fire before any deletion work touches the dataset.
    mock_get_dataset_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_data_proceeds_when_processing_completed():
    """Once processing completes, deletion must proceed past the guard."""
    dataset_id = uuid4()
    data_id = uuid4()
    user = _make_user(uuid4())
    dataset = _make_dataset(dataset_id, user.id)

    with (
        patch(
            "cognee.api.v1.datasets.datasets.get_authorized_dataset",
            new=AsyncMock(return_value=dataset),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.get_pipeline_status",
            new=AsyncMock(
                return_value={str(dataset_id): PipelineRunStatus.DATASET_PROCESSING_COMPLETED}
            ),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.get_dataset_data",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.set_database_global_context_variables"
        ) as mock_context,
        patch(
            "cognee.api.v1.datasets.datasets.delete_data_nodes_and_edges",
            new=AsyncMock(),
        ) as mock_delete_nodes,
    ):
        mock_context.return_value.__aenter__ = AsyncMock(return_value=None)
        mock_context.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await datasets.delete_data(dataset_id, data_id, user)

    assert result == {"status": "success"}
    mock_delete_nodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_data_proceeds_when_no_pipeline_status():
    """A dataset with no recorded pipeline run must not be blocked."""
    dataset_id = uuid4()
    user = _make_user(uuid4())
    dataset = _make_dataset(dataset_id, user.id)

    with (
        patch(
            "cognee.api.v1.datasets.datasets.get_authorized_dataset",
            new=AsyncMock(return_value=dataset),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.get_pipeline_status",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.get_dataset_data",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "cognee.api.v1.datasets.datasets.set_database_global_context_variables"
        ) as mock_context,
        patch(
            "cognee.api.v1.datasets.datasets.delete_data_nodes_and_edges",
            new=AsyncMock(),
        ),
    ):
        mock_context.return_value.__aenter__ = AsyncMock(return_value=None)
        mock_context.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await datasets.delete_data(dataset_id, uuid4(), user)

    assert result == {"status": "success"}
