import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from types import SimpleNamespace

from cognee.api.client import app
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture(scope="session")
def test_client():
    """Keep a single TestClient for the whole module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_default_user():
    """Mock default user for testing."""
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        email="test@example.com",
        is_active=True,
        tenant_id=str(uuid.uuid4()),
    )


@pytest.fixture
def client(test_client, mock_default_user):
    """Setup test client with mocked authentication."""
    async def override_get_authenticated_user():
        return mock_default_user

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


class TestGetTenantRolesEndpoint:
    """Tests for the GET /tenants/{tenant_id}/roles endpoint"""

    @patch("cognee.modules.users.permissions.methods.has_user_management_permission", new_callable=AsyncMock)
    @patch("cognee.modules.users.roles.methods.get_roles_in_tenant", new_callable=AsyncMock)
    def test_get_tenant_roles_success(
        self, mock_get_roles, mock_permission_check, client, mock_default_user
    ):
        """Test successful retrieval of all roles in a tenant"""
        tenant_id = str(uuid.uuid4())
        
        mock_permission_check.return_value = True
        
        mock_role_1 = SimpleNamespace(
            id=str(uuid.uuid4()),
            name="Admin",
            description="Administrator role with full access",
            user_count=2,
            dict=lambda: {
                "id": str(uuid.uuid4()),
                "name": "Admin",
                "description": "Administrator role with full access",
                "user_count": 2,
            }
        )

        mock_role_2 = SimpleNamespace(
            id=str(uuid.uuid4()),
            name="Viewer",
            description="Read-only access",
            user_count=5,
            dict=lambda: {
                "id": str(uuid.uuid4()),
                "name": "Viewer",
                "description": "Read-only access",
                "user_count": 5,
            }
        )
        
        mock_get_roles.return_value = [mock_role_1, mock_role_2]
        
        response = client.get(f"/api/v1/permissions/tenants/{tenant_id}/roles")
        
        assert response.status_code == 200
        data = response.json()
        assert "roles" in data
        assert len(data["roles"]) == 2
        assert data["roles"][0]["name"] == "Admin"
        assert data["roles"][0]["description"] == "Administrator role with full access"
        assert data["roles"][0]["user_count"] == 2
        assert data["roles"][1]["name"] == "Viewer"
        assert data["roles"][1]["description"] == "Read-only access"
        assert data["roles"][1]["user_count"] == 5

    @patch("cognee.modules.users.permissions.methods.has_user_management_permission", new_callable=AsyncMock)
    def test_get_tenant_roles_permission_denied(
        self, mock_permission_check, client, mock_default_user
    ):
        """Test that non-tenant owners cannot retrieve roles"""
        from cognee.modules.users.exceptions import PermissionDeniedError
        
        tenant_id = str(uuid.uuid4())
        
        mock_permission_check.side_effect = PermissionDeniedError(
            message="User is not authorized to manage users for this tenant"
        )
        
        response = client.get(f"/api/v1/permissions/tenants/{tenant_id}/roles")
        
        assert response.status_code == 403

    @patch("cognee.modules.users.permissions.methods.has_user_management_permission", new_callable=AsyncMock)
    def test_get_tenant_roles_tenant_not_found(
        self, mock_permission_check, client, mock_default_user
    ):
        """Test that request for non-existent tenant returns 404"""
        from cognee.modules.users.exceptions import TenantNotFoundError
        
        tenant_id = str(uuid.uuid4())
        
        mock_permission_check.side_effect = TenantNotFoundError(
            message=f"Could not find tenant: {tenant_id}"
        )
        
        response = client.get(f"/api/v1/permissions/tenants/{tenant_id}/roles")
        
        assert response.status_code == 404

    @patch("cognee.modules.users.permissions.methods.has_user_management_permission", new_callable=AsyncMock)
    @patch("cognee.modules.users.roles.methods.get_roles_in_tenant", new_callable=AsyncMock)
    def test_get_tenant_roles_empty_list(
        self, mock_get_roles, mock_permission_check, client, mock_default_user
    ):
        """Test retrieval when tenant has no roles"""
        tenant_id = str(uuid.uuid4())
        
        mock_permission_check.return_value = True
        mock_get_roles.return_value = []
        
        response = client.get(f"/api/v1/permissions/tenants/{tenant_id}/roles")
        
        assert response.status_code == 200
        data = response.json()
        assert "roles" in data
        assert len(data["roles"]) == 0

    @patch("cognee.modules.users.permissions.methods.has_user_management_permission", new_callable=AsyncMock)
    @patch("cognee.modules.users.roles.methods.get_roles_in_tenant", new_callable=AsyncMock)
    def test_get_tenant_roles_response_format(
        self, mock_get_roles, mock_permission_check, client, mock_default_user
    ):
        """Test that response includes all required role fields"""
        tenant_id = str(uuid.uuid4())
        role_id = str(uuid.uuid4())
        
        mock_permission_check.return_value = True
        
        mock_role = SimpleNamespace(
            id=role_id,
            name="TestRole",
            description="Test role description",
            user_count=3,
            dict=lambda: {
                "id": role_id,
                "name": "TestRole",
                "description": "Test role description",
                "user_count": 3,
            }
        )
        
        mock_get_roles.return_value = [mock_role]
        
        response = client.get(f"/api/v1/permissions/tenants/{tenant_id}/roles")
        
        assert response.status_code == 200
        data = response.json()
        role = data["roles"][0]
        
        # verify all required fields are present
        assert "id" in role
        assert "name" in role
        assert "description" in role
        assert "user_count" in role
        
        # verify field values
        assert role["name"] == "TestRole"
        assert role["description"] == "Test role description"
        assert role["user_count"] == 3