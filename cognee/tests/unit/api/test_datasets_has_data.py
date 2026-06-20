from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.datasets.datasets import datasets


@pytest.mark.asyncio
async def test_has_data_passes_user_object_to_authorization():
    """Regression test: datasets.has_data must pass the User object, not user.id.

    The bug passed ``user.id`` (a UUID) to ``get_authorized_dataset``, whose
    first parameter is a ``User``. Downstream ``get_authorized_existing_datasets``
    dereferences ``user.id``, so a UUID raised ``AttributeError`` and the public
    ``has_data`` API was broken. All sibling methods (list_data, empty_dataset,
    delete_data) correctly pass the user object.
    """
    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()
    fake_dataset = SimpleNamespace(id=dataset_id)

    with (
        patch(
            "cognee.api.v1.datasets.datasets.get_authorized_dataset",
            new=AsyncMock(return_value=fake_dataset),
        ) as mock_get_authorized_dataset,
        patch(
            "cognee.api.v1.datasets.datasets.has_dataset_data",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await datasets.has_data(dataset_id, user=user)

    assert result is True
    # First positional arg must be the User object itself, not user.id.
    first_arg = mock_get_authorized_dataset.call_args.args[0]
    assert first_arg is user
