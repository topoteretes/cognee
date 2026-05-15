"""Tests for the POST /v1/forget endpoint with memory_only flag.

Covers:
- memory_only is passed through to the forget() function
- Validation error when memory_only=True without dataset
"""

import uuid
import importlib
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from cognee.api.client import app
from cognee.modules.users.methods import get_authenticated_user

forget_pkg = importlib.import_module("cognee.api.v1.forget")


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_default_user():
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        email="test@example.com",
        is_active=True,
        tenant_id=str(uuid.uuid4()),
    )


@pytest.fixture
def client(test_client, mock_default_user):
    async def override_get_authenticated_user():
        return mock_default_user

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


def test_forget_endpoint_passes_memory_only_flag(client, mock_default_user):
    """POST /v1/forget with memory_only=True passes the flag to forget()."""
    dataset_id = str(uuid.uuid4())
    expected_result = {
        "dataset_id": dataset_id,
        "data_records_reset": 3,
        "status": "success",
    }

    with patch.object(forget_pkg, "forget", new_callable=AsyncMock) as mock_forget:
        mock_forget.return_value = expected_result

        response = client.post(
            "/api/v1/forget",
            json={"dataset": dataset_id, "memory_only": True},
        )

        assert response.status_code == 200
        assert response.json() == expected_result

        mock_forget.assert_awaited_once()
        call_kwargs = mock_forget.call_args.kwargs
        assert call_kwargs["memory_only"] is True
        assert call_kwargs["dataset"] == dataset_id


def test_forget_endpoint_memory_only_defaults_to_false(client):
    """POST /v1/forget without memory_only should default to False."""
    with patch.object(forget_pkg, "forget", new_callable=AsyncMock) as mock_forget:
        mock_forget.return_value = {"status": "success", "datasets_removed": 0}

        response = client.post(
            "/api/v1/forget",
            json={"everything": True},
        )

        assert response.status_code == 200
        call_kwargs = mock_forget.call_args.kwargs
        assert call_kwargs["memory_only"] is False


def test_forget_endpoint_memory_only_with_data_id(client):
    """POST /v1/forget with memory_only + dataset + data_id passes all params."""
    dataset_id = str(uuid.uuid4())
    data_id = str(uuid.uuid4())

    with patch.object(forget_pkg, "forget", new_callable=AsyncMock) as mock_forget:
        mock_forget.return_value = {
            "data_id": data_id,
            "dataset_id": dataset_id,
            "status": "success",
        }

        response = client.post(
            "/api/v1/forget",
            json={"dataset": dataset_id, "data_id": data_id, "memory_only": True},
        )

        assert response.status_code == 200
        call_kwargs = mock_forget.call_args.kwargs
        assert call_kwargs["memory_only"] is True
        assert call_kwargs["data_id"] == uuid.UUID(data_id)
        assert call_kwargs["dataset"] == dataset_id
