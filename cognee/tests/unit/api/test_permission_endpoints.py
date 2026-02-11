import os
import pytest
import uuid
import importlib
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient
from types import SimpleNamespace

os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
from cognee.api.client import app
from cognee.modules.users.methods import get_authenticated_user

gau_mod = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")

# Explicitly load the module by its full path
router_module = importlib.import_module("cognee.api.v1.permissions.routers.get_permissions_router")


@pytest.fixture
def client():
    """Create a test client with mocked authentication."""
    from cognee.api.client import app

    return TestClient(app)


@pytest.fixture
def mock_tenant_owner():
    """Mock tenant owner user."""
    return SimpleNamespace(id=uuid4(), email="owner@example.com", is_active=True, tenant_id=uuid4())


@pytest.mark.asyncio
@patch.object(router_module, "get_relational_engine")
@patch.object(router_module, "get_tenant")
def test_get_tenant_roles_success(mock_get_tenant, mock_get_engine, client, mock_tenant_owner):
    """Test successful role listing by tenant owner."""

    # Use the mock user's tenant_id to ensure ownership check passes
    tenant_id = mock_tenant_owner.tenant_id

    # Mock the tenant to pass ownership check
    mock_tenant = MagicMock()
    mock_tenant.owner_id = mock_tenant_owner.id
    mock_get_tenant.return_value = mock_tenant

    # Mock database session and query results
    mock_role = MagicMock()
    mock_role.id = uuid4()
    mock_role.name = "test_role"
    mock_role.description = "A test role"
    mock_role.users = []

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_role]

    mock_session_instance = AsyncMock()
    mock_session_instance.__aenter__.return_value.execute = AsyncMock(return_value=mock_result)

    mock_engine = MagicMock()
    mock_engine.get_async_session.return_value = mock_session_instance
    mock_get_engine.return_value = mock_engine

    async def override_get_authenticated_user():
        return mock_tenant_owner

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user

    try:
        response = client.get(f"/api/v1/permissions/tenants/{tenant_id}/roles")
        assert response.status_code == 200
        data = response.json()
        assert "roles" in data
        assert isinstance(data["roles"], list)
    finally:
        app.dependency_overrides.pop(get_authenticated_user, None)
