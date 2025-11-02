import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
from types import SimpleNamespace
import importlib
from cognee.api.client import app

gau_mod = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_user():
    user = Mock()
    user.id = "test-user-123"
    return user

@pytest.fixture
def mock_default_user():
    """Mock default user for testing."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="default@example.com",
        is_active=True,
        tenant_id=uuid.uuid4()
    )

@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_success(mock_get_default_user, client, mock_default_user):
    """Test successful ontology upload"""
    mock_get_default_user.return_value = mock_default_user
    ontology_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/api/v1/ontologies",
        files={"ontology_file": ("test.owl", ontology_content)},
        data={"ontology_key": unique_key, "description": "Test"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ontology_key"] == unique_key
    assert "uploaded_at" in data

@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_invalid_file(mock_get_default_user, client, mock_default_user):
    """Test 400 response for non-.owl files"""
    mock_get_default_user.return_value = mock_default_user
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/api/v1/ontologies",
        files={"ontology_file": ("test.txt", b"not xml")},
        data={"ontology_key": unique_key}
    )
    assert response.status_code == 400

@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_missing_data(mock_get_default_user, client, mock_default_user):
    """Test 400 response for missing file or key"""
    mock_get_default_user.return_value = mock_default_user
    # Missing file
    response = client.post("/api/v1/ontologies", data={"ontology_key": "test"})
    assert response.status_code == 400

    # Missing key
    response = client.post("/api/v1/ontologies", files={"ontology_file": ("test.owl", b"xml")})
    assert response.status_code == 400

@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_unauthorized(mock_get_default_user, client, mock_default_user):
    """Test behavior when default user is provided (no explicit authentication)"""
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
    mock_get_default_user.return_value = mock_default_user
    response = client.post(
        "/api/v1/ontologies",
        files={"ontology_file": ("test.owl", b"<rdf></rdf>")},
        data={"ontology_key": unique_key}
    )

    # The current system provides a default user when no explicit authentication is given
    # This test verifies the system works with conditional authentication
    assert response.status_code == 200
    data = response.json()
    assert data["ontology_key"] == unique_key
    assert "uploaded_at" in data