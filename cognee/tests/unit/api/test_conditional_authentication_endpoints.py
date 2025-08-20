import os
import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient
from types import SimpleNamespace

from cognee.api.client import app


class TestConditionalAuthenticationEndpoints:
    """Test that API endpoints work correctly with conditional authentication."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_default_user(self):
        """Mock default user for testing."""
        return SimpleNamespace(
            id=uuid4(),
            email="default@example.com", 
            is_active=True,
            tenant_id=uuid4()
        )
    
    @pytest.fixture
    def mock_authenticated_user(self):
        """Mock authenticated user for testing."""
        from cognee.modules.users.models import User
        return User(
            id=uuid4(),
            email="auth@example.com",
            hashed_password="hashed",
            is_active=True,
            is_verified=True,
            tenant_id=uuid4()
        )

    def test_health_endpoint_no_auth_required(self, client):
        """Test that health endpoint works without authentication."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # 503 is also acceptable for health checks
    
    def test_root_endpoint_no_auth_required(self, client):
        """Test that root endpoint works without authentication."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello, World, I am alive!"}
    
    @patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"})
    def test_openapi_schema_no_global_security(self, client):
        """Test that OpenAPI schema doesn't require global authentication."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        
        # Should not have global security requirement
        global_security = schema.get("security", [])
        assert global_security == []
        
        # But should still have security schemes defined
        security_schemes = schema.get("components", {}).get("securitySchemes", {})
        assert "BearerAuth" in security_schemes
        assert "CookieAuth" in security_schemes
    
    @patch.dict(os.environ, {"REQUIRE_AUTHENTICATION": "false"})
    def test_add_endpoint_with_conditional_auth(self, client, mock_default_user):
        """Test add endpoint works with conditional authentication."""
        with patch('cognee.modules.users.methods.get_conditional_authenticated_user.get_default_user') as mock_get_default:
            with patch('cognee.api.v1.add.add') as mock_cognee_add:
                mock_get_default.return_value = mock_default_user
                mock_cognee_add.return_value = MagicMock(
                    model_dump=lambda: {"status": "success", "pipeline_run_id": str(uuid4())}
                )
                
                # Test file upload without authentication
                files = {"data": ("test.txt", b"test content", "text/plain")}
                form_data = {"datasetName": "test_dataset"}
                
                response = client.post("/api/v1/add", files=files, data=form_data)
                
                # Should succeed (not 401) 
                assert response.status_code != 401
                
                # Should have called get_default_user for anonymous request
                mock_get_default.assert_called()
    
    def test_conditional_authentication_works_with_current_environment(self, client):
        """Test that conditional authentication works with the current environment setup."""
        # Since REQUIRE_AUTHENTICATION defaults to "false", we expect endpoints to work without auth
        # This tests the actual integration behavior
        
        with patch('cognee.modules.users.methods.get_conditional_authenticated_user.get_default_user') as mock_get_default:
            mock_default_user = SimpleNamespace(id=uuid4(), email="default@example.com", is_active=True, tenant_id=uuid4())
            mock_get_default.return_value = mock_default_user
            
            files = {"data": ("test.txt", b"test content", "text/plain")}
            form_data = {"datasetName": "test_dataset"}
            
            response = client.post("/api/v1/add", files=files, data=form_data)
            
            # Should not return 401 (authentication not required with default environment)
            assert response.status_code != 401
            
            # Should have called get_default_user for anonymous request
            mock_get_default.assert_called()
    
    def test_authenticated_request_uses_user(self, client, mock_authenticated_user):
        """Test that authenticated requests use the authenticated user, not default user."""
        with patch('cognee.modules.users.methods.get_conditional_authenticated_user.get_default_user') as mock_get_default:
            with patch('cognee.api.v1.add.add') as mock_cognee_add:
                # Mock successful authentication - this would normally be handled by FastAPI Users
                # but we're testing the conditional logic
                mock_cognee_add.return_value = MagicMock(
                    model_dump=lambda: {"status": "success", "pipeline_run_id": str(uuid4())}
                )
                
                # Simulate authenticated request by directly testing the conditional function
                from cognee.modules.users.methods.get_conditional_authenticated_user import get_conditional_authenticated_user
                
                async def test_logic():
                    # When user is provided (authenticated), should not call get_default_user
                    result = await get_conditional_authenticated_user(user=mock_authenticated_user)
                    assert result == mock_authenticated_user
                    mock_get_default.assert_not_called()
                
                # Run the async test
                import asyncio
                asyncio.run(test_logic())


class TestConditionalAuthenticationBehavior:
    """Test the behavior of conditional authentication across different endpoints."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.parametrize("endpoint,method", [
        ("/api/v1/search", "GET"),
        ("/api/v1/datasets", "GET"),
    ])
    def test_get_endpoints_work_without_auth(self, client, endpoint, method, mock_default_user):
        """Test that GET endpoints work without authentication (with current environment)."""
        with patch('cognee.modules.users.methods.get_conditional_authenticated_user.get_default_user') as mock_get_default:
            mock_get_default.return_value = mock_default_user
            
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            # Should not return 401 Unauthorized (authentication is optional by default)
            assert response.status_code != 401
            
            # May return other errors due to missing data/config, but not auth errors
            if response.status_code >= 400:
                # Check that it's not an authentication error
                try:
                    error_detail = response.json().get("detail", "")
                    assert "authenticate" not in error_detail.lower()
                    assert "unauthorized" not in error_detail.lower()
                except:
                    pass  # If response is not JSON, that's fine
    
    def test_settings_endpoint_integration(self, client, mock_default_user):
        """Test that settings endpoint integration works with conditional authentication."""
        with patch('cognee.modules.users.methods.get_conditional_authenticated_user.get_default_user') as mock_get_default:
            with patch('cognee.modules.settings.get_settings.get_llm_config') as mock_llm_config:
                with patch('cognee.modules.settings.get_settings.get_vectordb_config') as mock_vector_config:
                    mock_get_default.return_value = mock_default_user
                    
                    # Mock configurations to avoid validation errors
                    mock_llm_config.return_value = SimpleNamespace(
                        llm_provider="openai",
                        llm_model="gpt-4o", 
                        llm_endpoint=None,
                        llm_api_version=None,
                        llm_api_key="test_key_1234567890"
                    )
                    
                    mock_vector_config.return_value = SimpleNamespace(
                        vector_db_provider="lancedb",
                        vector_db_url="localhost:5432",  # Must be string, not None
                        vector_db_key="test_vector_key"
                    )
                    
                    response = client.get("/api/v1/settings")
                    
                    # Should not return 401 (authentication works)
                    assert response.status_code != 401
                    
                    # Should have called get_default_user for anonymous request
                    mock_get_default.assert_called()


class TestConditionalAuthenticationErrorHandling:
    """Test error handling in conditional authentication."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_get_default_user_fails(self, client):
        """Test behavior when get_default_user fails (with current environment)."""
        with patch('cognee.modules.users.methods.get_conditional_authenticated_user.get_default_user') as mock_get_default:
            mock_get_default.side_effect = Exception("Database connection failed")
            
            # The error should propagate - either as a 500 error or as an exception
            files = {"data": ("test.txt", b"test content", "text/plain")}
            form_data = {"datasetName": "test_dataset"}
            
            # Test that the exception is properly converted to HTTP 500
            response = client.post("/api/v1/add", files=files, data=form_data)
            
            # Should return HTTP 500 Internal Server Error when get_default_user fails
            assert response.status_code == 500
            
            # Check that the error message is informative
            error_detail = response.json().get("detail", "")
            assert "Failed to create default user" in error_detail
            assert "Database connection failed" in error_detail
            
            # Most importantly, verify that get_default_user was called (the conditional auth is working)
            mock_get_default.assert_called()
    
    def test_current_environment_configuration(self):
        """Test that current environment configuration is working properly."""
        # This tests the actual module state without trying to change it
        from cognee.modules.users.methods.get_conditional_authenticated_user import REQUIRE_AUTHENTICATION
        
        # Should be a boolean value (the parsing logic works)
        assert isinstance(REQUIRE_AUTHENTICATION, bool)
        
        # In default environment, should be False
        assert REQUIRE_AUTHENTICATION == False


# Fixtures for reuse across test classes
@pytest.fixture
def mock_default_user():
    """Mock default user for testing."""
    return SimpleNamespace(
        id=uuid4(),
        email="default@example.com",
        is_active=True,
        tenant_id=uuid4()
    )

@pytest.fixture  
def mock_authenticated_user():
    """Mock authenticated user for testing."""
    from cognee.modules.users.models import User
    return User(
        id=uuid4(), 
        email="auth@example.com",
        hashed_password="hashed",
        is_active=True,
        is_verified=True,
        tenant_id=uuid4()
    )
