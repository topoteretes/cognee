import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from cognee.api.v1.datasets.datasets import datasets

@pytest.mark.asyncio
async def test_datasets_has_data_passes_user_object():
    mock_user = MagicMock()
    mock_user.id = uuid4()
    mock_dataset = MagicMock()
    mock_dataset.id = uuid4()

    with (
        patch("cognee.api.v1.datasets.datasets.get_default_user", AsyncMock(return_value=mock_user)) as mock_get_default_user,
        patch("cognee.api.v1.datasets.datasets.get_authorized_dataset", AsyncMock(return_value=mock_dataset)) as mock_get_auth_ds,
        patch("cognee.api.v1.datasets.datasets.has_dataset_data", AsyncMock(return_value=True)) as mock_has_ds_data,
    ):
        result = await datasets.has_data("some_dataset_id")
        
        assert result is True
        mock_get_default_user.assert_awaited_once()
        # Verify that the first argument passed to get_authorized_dataset is the user object, not user.id
        mock_get_auth_ds.assert_awaited_once_with(mock_user, "some_dataset_id")
        mock_has_ds_data.assert_awaited_once_with(mock_dataset.id)
