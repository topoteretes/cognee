import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient
from types import SimpleNamespace
import importlib


# Fixtures for reuse across test classes
@pytest.fixture
def mock_default_user():
    """Mock default user for testing."""
    return SimpleNamespace(
        id=uuid4(), email="default@example.com", is_active=True, tenant_id=uuid4()
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
        tenant_id=uuid4(),
    )


# To turn off authentication we need to set the environment variable before importing the module
# Also both require_authentication and backend access control must be false
os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
gau_mod = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")


class TestConditionalAuthenticationEndpoints:
    """Test that API endpoints work correctly with conditional authentication."""

    @pytest.fixture
    def client(self):
        from cognee.api.client import app

        """Create a test client."""
        return TestClient(app)

    def test_health_endpoint_no_auth_required(self, client):
        """Test that health endpoint works without authentication."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # 503 is also acceptable for health checks

    def test_root_endpoint_no_auth_required(self, client):
        """Test that root endpoint works without authentication."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello, World, I am alive!"}

    @patch(
        "cognee.api.client.REQUIRE_AUTHENTICATION",
        False,
    )
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

    @patch("cognee.api.v1.add.add")
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    @patch(
        "cognee.api.client.REQUIRE_AUTHENTICATION",
        False,
    )
    def test_add_endpoint_with_conditional_auth(
        self, mock_get_default_user, mock_add, client, mock_default_user
    ):
        """Test add endpoint works with conditional authentication."""
        mock_get_default_user.return_value = mock_default_user
        mock_add.return_value = MagicMock(
            model_dump=lambda: {"status": "success", "pipeline_run_id": str(uuid4())}
        )

        # Test file upload without authentication
        files = {"data": ("test.txt", b"test content", "text/plain")}
        form_data = {"datasetName": "test_dataset"}

        response = client.post("/api/v1/add", files=files, data=form_data)

        assert mock_get_default_user.call_count == 1

        # Core test: authentication is not required (should not get 401)
        assert response.status_code != 401

    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    @patch(
        "cognee.api.client.REQUIRE_AUTHENTICATION",
        False,
    )
    def test_conditional_authentication_works_with_current_environment(
        self, mock_get_default_user, client
    ):
        """Test that conditional authentication works with the current environment setup."""
        # Since REQUIRE_AUTHENTICATION defaults to "false", we expect endpoints to work without auth
        # This tests the actual integration behavior

        mock_get_default_user.return_value = SimpleNamespace(
            id=uuid4(), email="default@example.com", is_active=True, tenant_id=uuid4()
        )

        files = {"data": ("test.txt", b"test content", "text/plain")}
        form_data = {"datasetName": "test_dataset"}

        response = client.post("/api/v1/add", files=files, data=form_data)

        assert mock_get_default_user.call_count == 1

        # Core test: authentication is not required (should not get 401)
        assert response.status_code != 401
        # Note: This test verifies conditional authentication works in the current environment


class TestConditionalAuthenticationBehavior:
    """Test the behavior of conditional authentication across different endpoints."""

    @pytest.fixture
    def client(self):
        from cognee.api.client import app

        return TestClient(app)

    @pytest.mark.parametrize(
        "endpoint,method",
        [
            ("/api/v1/search", "GET"),
            ("/api/v1/datasets", "GET"),
        ],
    )
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    def test_get_endpoints_work_without_auth(
        self, mock_get_default, client, endpoint, method, mock_default_user
    ):
        """Test that GET endpoints work without authentication (with current environment)."""
        mock_get_default.return_value = mock_default_user

        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST":
            response = client.post(endpoint, json={})

        assert mock_get_default.call_count == 1

        # Should not return 401 Unauthorized (authentication is optional by default)
        assert response.status_code != 401

        # May return other errors due to missing data/config, but not auth errors
        if response.status_code >= 400:
            # Check that it's not an authentication error
            try:
                error_detail = response.json().get("detail", "")
                assert "authenticate" not in error_detail.lower()
                assert "unauthorized" not in error_detail.lower()
            except Exception:
                pass  # If response is not JSON, that's fine

    gsm_mod = importlib.import_module("cognee.modules.settings.get_settings")

    @patch.object(gsm_mod, "get_vectordb_config")
    @patch.object(gsm_mod, "get_llm_config")
    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    def test_settings_endpoint_integration(
        self, mock_get_default, mock_llm_config, mock_vector_config, client, mock_default_user
    ):
        """Test that settings endpoint integration works with conditional authentication."""
        mock_get_default.return_value = mock_default_user

        # Mock configurations to avoid validation errors
        mock_llm_config.return_value = SimpleNamespace(
            llm_provider="openai",
            llm_model="gpt-4o",
            llm_endpoint=None,
            llm_api_version=None,
            llm_api_key="test_key_1234567890",
        )

        mock_vector_config.return_value = SimpleNamespace(
            vector_db_provider="lancedb",
            vector_db_url="localhost:5432",  # Must be string, not None
            vector_db_key="test_vector_key",
        )

        response = client.get("/api/v1/settings")

        assert mock_get_default.call_count == 1

        # Core test: authentication is not required (should not get 401)
        assert response.status_code != 401
        # Note: This test verifies conditional authentication works for settings endpoint


class TestConditionalAuthenticationErrorHandling:
    """Test error handling in conditional authentication."""

    @pytest.fixture
    def client(self):
        from cognee.api.client import app

        return TestClient(app)

    @patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
    def test_get_default_user_fails(self, mock_get_default, client):
        """Test behavior when get_default_user fails (with current environment)."""
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
        # The exact error message may vary depending on the actual database connection
        # The important thing is that we get a 500 error when user creation fails

    def test_current_environment_configuration(self, client):
        """Test that current environment configuration is working properly."""
        # This tests the actual module state without trying to change it
        from cognee.modules.users.methods.get_authenticated_user import (
            REQUIRE_AUTHENTICATION,
        )

        # Should be a boolean value (the parsing logic works)
        assert isinstance(REQUIRE_AUTHENTICATION, bool)

        # In default environment, should be False
        assert not REQUIRE_AUTHENTICATION
