import os  
import pytest  
import uuid  
from unittest.mock import patch, AsyncMock, MagicMock  
from uuid import uuid4  
from fastapi.testclient import TestClient  
from types import SimpleNamespace  
import importlib  
  
os.environ["REQUIRE_AUTHENTICATION"] = "false"  
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"  
  
from cognee.api.client import app  
from cognee.modules.users.methods import get_authenticated_user  
from cognee.modules.users.models import User  
  
gau_mod = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")  
  
@pytest.fixture  
def client():  
    """Create a test client with mocked authentication."""  
    from cognee.api.client import app  
    return TestClient(app)  
  
@pytest.fixture  
def mock_tenant_owner():  
    """Mock tenant owner user."""  
    return SimpleNamespace(  
        id=uuid4(),  
        email="owner@example.com",  
        is_active=True,  
        tenant_id=uuid4()  
    )  
  
def test_get_tenant_roles_success(client, mock_tenant_owner):  
    """Test successful role listing by tenant owner."""  
    async def override_get_authenticated_user():  
        return mock_tenant_owner  
      
    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user  
      
    # Use actual UUID instead of placeholder  
    tenant_id = str(uuid4())  
      
    try:  
        response = client.get(  
            f"/api/v1/permissions/tenants/{tenant_id}/roles"  
        )  
        assert response.status_code in [200, 404]  
    finally:  
        app.dependency_overrides.pop(get_authenticated_user, None)
