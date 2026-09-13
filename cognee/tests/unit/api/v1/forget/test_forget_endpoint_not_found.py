"""Regression tests for typed errors from POST /v1/forget (issue #5013).

Before the fix, an unknown dataset name crashed forget() with
`AttributeError: 'NoneType' object has no attribute 'id'` and the router's
catch-all turned every typed error (including PermissionDeniedError, which
carries status_code=403) into `500 {"error": "An error occurred during deletion."}`.
Now `_resolve_dataset_id` raises DatasetNotFoundError for an unresolvable name
and the router re-raises CogneeApiError so the global handler renders its code.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from cognee.api.client import app
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user

forget_pkg = importlib.import_module("cognee.api.v1.forget")


@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="test@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


def test_unknown_dataset_name_returns_404(client):
    """forget(dataset=<unknown name>) must surface as 404, not a 500 NoneType crash."""
    with (
        patch.object(forget_pkg, "forget", new_callable=AsyncMock) as mock_forget,
        patch("cognee.api.v1.forget.routers.get_forget_router.send_telemetry"),
    ):
        mock_forget.side_effect = DatasetNotFoundError(
            message="Dataset 'does-not-exist' not found or not accessible."
        )

        response = client.post("/api/v1/forget", json={"dataset": "does-not-exist"})

    assert response.status_code == 404
    assert "does-not-exist" in response.text
    assert "DatasetNotFoundError" in response.text


def test_permission_denied_no_longer_masked_as_500(client):
    """PermissionDeniedError carries status_code=403 and must reach the client as 403."""
    with (
        patch.object(forget_pkg, "forget", new_callable=AsyncMock) as mock_forget,
        patch("cognee.api.v1.forget.routers.get_forget_router.send_telemetry"),
    ):
        mock_forget.side_effect = PermissionDeniedError(
            message="You do not have permission to delete this dataset."
        )

        response = client.post("/api/v1/forget", json={"datasetId": str(uuid.uuid4())})

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_resolve_dataset_id_raises_typed_error_for_unknown_name():
    """The lookup site itself: a None dataset must raise DatasetNotFoundError, not crash."""
    from cognee.api.v1.forget.forget import _resolve_dataset_id

    user = SimpleNamespace(id=uuid.uuid4(), email="test@example.com")
    with (
        patch(
            "cognee.modules.data.methods.get_authorized_dataset_by_name",
            new_callable=AsyncMock,
            return_value=None,
        ),
        pytest.raises(DatasetNotFoundError) as exc_info,
    ):
        await _resolve_dataset_id("does-not-exist", user)

    assert "does-not-exist" in str(exc_info.value)
    assert exc_info.value.status_code == 404
